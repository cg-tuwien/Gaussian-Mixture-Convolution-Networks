import pathlib
import typing
import time
import datetime
import sys
import traceback

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.tensorboard

import gm
from gm import MixtureReLUandBias
from gm import Mixture
import config
import madam_imagetools

COVARIANCE_MIN = 0.0001


class Net(nn.Module):
    def __init__(self,
                 g_layer_sizes: typing.List,
                 fully_layer_sizes: typing.List,
                 n_output_gaussians: int,
                 name: str = "",
                 n_dims: int = 2,
                 n_agrs: int = 4,
                 batch_norm: bool = False):
        super(Net, self).__init__()
        self.n_dims = n_dims
        self.n_agrs = n_agrs
        self.batch_norm = batch_norm
        self.n_output_gaussians = n_output_gaussians
        # n * (1 for weights, DIMS for positions, trimat_size(DIMS) for the triangle cov matrix) +1 for the bias
        n_inputs_per_gaussian = 1 + n_dims + n_dims * n_dims
        # n_inputs = N_INPUT_GAUSSIANS * (1 + DIMS + mat_tools.trimat_size(DIMS)) + 1
        # and we want to output A, so that C = A @ A.T() + 0.01 * identity() is the cov matrix
        n_outputs_per_gaussian = 1 + n_dims + n_dims * n_dims
        # n_outputs = N_OUTPUT_GAUSSIANS * n_outputs_per_gaussian

        last_layer_size = n_inputs_per_gaussian
        self.per_g_layers = nn.ModuleList()
        self.per_g_batch_norms = nn.ModuleList()
        for s in g_layer_sizes[:-1]:
            self.per_g_layers.append(nn.Conv1d(last_layer_size, s, kernel_size=1, stride=1, groups=1))
            if self.batch_norm:
                self.per_g_batch_norms.append(nn.BatchNorm1d(s))
            last_layer_size = s
        self.per_g_output_layer = nn.Conv1d(last_layer_size, g_layer_sizes[-1], kernel_size=1, stride=1, groups=1)
        last_layer_size = g_layer_sizes[-1]

        self.fully_layers = nn.ModuleList()
        self.fully_batch_norms = nn.ModuleList()

        assert last_layer_size % (self.n_output_gaussians * self.n_agrs) == 0
        last_layer_size = last_layer_size // self.n_output_gaussians + 1
        for s in fully_layer_sizes:
            self.fully_layers.append(nn.Conv1d(last_layer_size, s, kernel_size=1, stride=1, groups=1))
            if self.batch_norm:
                self.fully_batch_norms.append(nn.BatchNorm1d(s))
            last_layer_size = s

        self.output_layer = nn.Conv1d(last_layer_size, n_outputs_per_gaussian, kernel_size=1, stride=1, groups=1)

        self.name = f"fit_gm_net_woio{'_bn' if self.batch_norm else ''}_a{self.n_agrs}"
        self.name += name + "_g"
        for s in g_layer_sizes:
            self.name += f"_{s}"
        self.name += "__f"
        for s in fully_layer_sizes:
            self.name += f"_{s}"

        self.storage_path = config.data_base_path / "weights" / self.name

    def save(self):
        print(f"gm_fitting.Net: saving to {self.storage_path}")
        torch.save(self.state_dict(), self.storage_path)

    def load(self, strict: bool = False):
        print(f"gm_fitting.Net: trying to load {self.storage_path}")
        if pathlib.Path(self.storage_path).is_file():
            state_dict = torch.load(self.storage_path)
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)
            # assert len(missing_keys) == 0
            # assert len(unexpected_keys) == 0
            print(f"gm_fitting.Net: loaded (missing: {missing_keys}, unexpected: {unexpected_keys}")
            return True
        else:
            print("gm_fitting.Net: not found")
            return False

    def device(self):
        return self.output_layer.bias.device

    def forward(self, data_in: MixtureReLUandBias, learning: bool = True) -> Mixture:
        n_batch = data_in.mixture.n_batch()
        n_layers = data_in.mixture.n_layers()
        n_input_components = data_in.mixture.n_components()
        n_dims = data_in.mixture.n_dimensions()

        data_normalised, normalisation_factors = gm.normalise(data_in)

        cov_data = data_normalised.mixture.covariances
        # view weights, positions and covariances as one data vector
        x = torch.cat((data_normalised.mixture.weights.view(n_batch * n_layers, n_input_components, 1),
                       data_normalised.mixture.positions.view(n_batch * n_layers, n_input_components, 1),
                       cov_data.view(n_batch * n_layers, n_input_components, n_dims * n_dims)), dim=2)
        # component should be the last dimension for conv1d to work
        x = x.transpose(1, 2)

        for i, layer in enumerate(self.per_g_layers):
            x = layer(x)
            if self.batch_norm:
                x = self.per_g_batch_norms[i](x)
            x = F.leaky_relu(x)
        x = self.per_g_output_layer(x)

        n_agrs = self.n_agrs
        agrs_list = []
        n_out = x.shape[1]
        x_sum = torch.sum(x[:,                 0:(n_out//n_agrs)*1, :], dim=2).view(n_batch * n_layers, -1, 1)
        agrs_list.append(x_sum)
        if n_agrs >= 4:
            x_max = torch.max(x[:, (n_out//n_agrs)*3:(n_out//n_agrs)*4, :], dim=2).values.view(n_batch * n_layers, -1, 1)
            agrs_list.append(x_max)
        if n_agrs >= 3:
            x_var = torch.var(x[:, (n_out//n_agrs)*2:(n_out//n_agrs)*3, :], dim=2, unbiased=False).view(n_batch * n_layers, -1, 1)
            agrs_list.append(x_var)
        if n_agrs >= 2:
            x_prd = torch.abs(x[:, (n_out//n_agrs)*1:(n_out//n_agrs)*2, :] + 1)
            # old, explods when switchng from n_gaussians=10 to 100
            # x_prd = torch.prod(x_prd, dim=2).view(n_layers, -1, 1)

            x_prd = torch.max(x_prd, torch.tensor(0.01, dtype=torch.float32, device=x.device))
            x_prd = torch.mean(torch.log(x_prd), dim=2).view(n_batch * n_layers, -1, 1)
            x_prd = torch.min(x_prd, torch.tensor(10, dtype=torch.float32, device=x.device))
            x_prd = torch.exp(x_prd)
            agrs_list.append(x_prd)

        x = torch.cat(agrs_list, dim=2).reshape(n_batch * n_layers, -1)
        latent_vector = x

        # x is batch size x final g layer size now
        x = x.view(n_batch * n_layers, -1, self.n_output_gaussians)
        x = torch.cat((data_normalised.bias.view(n_batch * n_layers, 1, 1).expand(n_batch * n_layers, 1, self.n_output_gaussians), x), dim=1)

        for i, layer in enumerate(self.fully_layers):
            x = layer(x)
            if self.batch_norm:
                x = self.fully_batch_norms[i](x)
            x = F.leaky_relu(x)
            # x = self.batch_norms[i](x.view(-1, 1))
            i += 1

        x = self.output_layer(x)
        x = x.transpose(1, 2)

        weights = x[:, :, 0]
        positions = x[:, :, 1:(self.n_dims + 1)]
        # we are learning A, so that C = A @ A.T() + 0.01 * identity() is the resulting cov matrix
        A = x[:, :, (self.n_dims + 1):].view(n_batch * n_layers, -1, self.n_dims, self.n_dims)
        C = A @ A.transpose(2, 3) + torch.eye(self.n_dims, self.n_dims, dtype=torch.float32, device=self.device()) * COVARIANCE_MIN
        covariances = C

        try:
            normalised_out = gm.Mixture(weights.view(n_batch, n_layers, n_input_components),
                                        positions.view(n_batch, n_layers, n_input_components, n_dims),
                                        covariances.view(n_batch, n_layers, n_input_components, n_dims, n_dims))
        except AssertionError:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb) # Fixed format
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]

            print('An error occurred on line {} in statement {}'.format(line, text))
            exit(1)
        return gm.de_normalise(normalised_out, normalisation_factors), latent_vector


class Trainer:
    def __init__(self, net: Net, n_training_samples: int = 50 * 50, learning_rate: float = 0.001, save_weights: bool = False, testing_mode: bool = False):
        self.net = net
        self.n_training_samples = n_training_samples
        self.learning_rate = learning_rate
        self.save_weights = save_weights
        self.testing_mode = testing_mode
        self.criterion = nn.MSELoss()
        self.optimiser = optim.Adam(net.parameters(), lr=learning_rate)
        self.tensor_board_writer = torch.utils.tensorboard.SummaryWriter(config.data_base_path / 'tensorboard' / f'{self.net.name}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
        self.tensor_board_graph_written = False

    def log_image(self, tag: str, image: typing.Sequence[torch.Tensor], epoch: int, clamp: typing.Sequence[float]):
        image = image.detach().t().cpu().numpy()
        image = madam_imagetools.colour_mapped(image, clamp[0], clamp[1])
        self.tensor_board_writer.add_image(tag, image, epoch, dataformats='HWC')

    def log_images(self, tag: str, images: typing.Sequence[torch.Tensor], epoch: int, clamp: typing.Sequence[float]):
        for i in range(len(images)):
            images[i] = images[i].detach().t().cpu().numpy()
            images[i] = madam_imagetools.colour_mapped(images[i], clamp[0], clamp[1])
            images[i] = images[i][:, :, 0:3]
            images[i] = images[i].reshape(1, images[i].shape[0], images[i].shape[1], 3)
        images = np.concatenate(images, axis=0)
        self.tensor_board_writer.add_image(tag, images, epoch, dataformats='NHWC')

    def train_on(self, data_in: gm.MixtureReLUandBias, epoch: int):
        data_in = data_in.detach()
        data_in, _ = gm.normalise(data_in)  # we normalise twice, but that shouldn't hurt (but performance). normalisation here is needed due to regularisation
        batch_size = data_in.mixture.n_batch()
        n_layers = data_in.mixture.n_layers()
        batch_start_time = time.perf_counter()
        self.optimiser.zero_grad()

        sampling_positions = torch.rand((1, 1, self.n_training_samples, data_in.mixture.n_dimensions()), dtype=torch.float32, device=self.net.device()) * 3 - 1.5
        target_sampling_values = data_in.evaluate_few_xes(sampling_positions)

        network_start_time = time.perf_counter()
        output_gm, latent_vector = self.net(data_in)
        network_time = time.perf_counter() - network_start_time

        eval_start_time = time.perf_counter()
        output_gm_sampling_values = output_gm.evaluate_few_xes(sampling_positions)
        criterion = self.criterion(output_gm_sampling_values, target_sampling_values) * 2

        # the network was moving gaussians out of the sampling radius
        p = (output_gm.positions.abs() - 1)
        p = p.where(p > torch.zeros(1, device=p.device), torch.zeros(1, device=p.device))
        regularisation = p.sum()

        eval_time = time.perf_counter() - eval_start_time

        backward_start_time = time.perf_counter()
        loss = criterion + regularisation
        loss.backward()
        backward_time = time.perf_counter() - backward_start_time

        self.optimiser.step()

        info = (f"gm_fitting.Trainer: epoch = {epoch}:"
                f"batch loss {loss.item():.4f} (crit: {criterion.item()} (rest is regularisation)), "
                f"batch time = {time.perf_counter() - batch_start_time :.2f}s, "
                f"size = {batch_size}, "
                f"(forward: {network_time :.2f}s (per layer: {network_time / batch_size :.4f}s), eval: {eval_time :.3f}s, backward: {backward_time :.4f}s) ")

        # if not self.tensor_board_graph_written:
        #     self.tensor_board_graph_written = True
        #     self.tensor_board_writer.add_graph(self.net, data_in)
        self.tensor_board_writer.add_scalar("0. batch_loss", loss.item(), epoch)
        self.tensor_board_writer.add_scalar("1. criterion", criterion.item(), epoch)
        # self.tensor_board_writer.add_scalar("3. regularisation_aggr_prod", regularisation_aggr_prod.item(), epoch)
        self.tensor_board_writer.add_scalar("4. whole time", time.perf_counter() - batch_start_time, epoch)
        self.tensor_board_writer.add_scalar("5. network_time", network_time, epoch)
        self.tensor_board_writer.add_scalar("6. eval_time", eval_time, epoch)
        self.tensor_board_writer.add_scalar("7. backward_time", backward_time, epoch)

        if (epoch % 10 == 0 and epoch < 100) or (epoch % 100 == 0 and epoch < 1000) or (epoch % 1000 == 0 and epoch < 10000) or (epoch % 10000 == 0):
            image_size = 128
            xv, yv = torch.meshgrid([torch.arange(-1.0, 1.0, 2 / image_size, dtype=torch.float, device=data_in.device()),
                                     torch.arange(-1.0, 1.0, 2 / image_size, dtype=torch.float, device=data_in.device())])
            xes = torch.cat((xv.reshape(-1, 1), yv.reshape(-1, 1)), 1).view(1, -1, 2).expand(batch_size, -1, 2)
            image_target = data_in.evaluate_few_xes(xes).view(-1, image_size, image_size)
            n_shown_images = 10
            fitted_mixture_image = output_gm.detach().evaluate_few_xes(xes).view(-1, image_size, image_size)
            self.log_images(f"target_image_mixture",
                            [image_target[:n_shown_images, :, :].transpose(0, 1).reshape(image_size, -1),
                             fitted_mixture_image[:n_shown_images, :, :].transpose(0, 1).reshape(image_size, -1)],
                            epoch, [-0.5, 2])
            self.log_image("latent_space", latent_vector.detach(), epoch, (-5, 5))

        print(info)
        if self.save_weights:
            self.net.save()
            f = open(config.data_base_path / "weights" / (self.net.name + "_loss"), "w")
            f.write(info)
            f.close()
