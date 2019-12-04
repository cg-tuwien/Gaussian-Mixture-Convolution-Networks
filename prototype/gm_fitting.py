import pathlib
import typing
import time

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gm
from gm import MixtureReLUandBias
from gm import Mixture

COVARIANCE_MIN = 0.0001


class Net(nn.Module):
    def __init__(self,
                 g_layer_sizes: typing.List,
                 fully_layer_sizes: typing.List,
                 n_output_gaussians: int,
                 name: str = "",
                 n_dims: int = 2,
                 output_image_width: int = 0,
                 output_image_height: int = 0):
        super(Net, self).__init__()
        self.n_dims = n_dims
        self.n_output_gaussians = n_output_gaussians
        # n * (1 for weights, DIMS for positions, trimat_size(DIMS) for the triangle cov matrix) +1 for the bias
        n_inputs_per_gaussian = 1 + n_dims + n_dims * n_dims
        # n_inputs = N_INPUT_GAUSSIANS * (1 + DIMS + mat_tools.trimat_size(DIMS)) + 1
        # and we want to output A, so that C = A @ A.T() + 0.01 * identity() is the cov matrix
        n_outputs_per_gaussian = 1 + n_dims + n_dims * n_dims
        # n_outputs = N_OUTPUT_GAUSSIANS * n_outputs_per_gaussian

        last_layer_size = n_inputs_per_gaussian
        self.per_g_layers = nn.ModuleList()
        for s in g_layer_sizes:
            self.per_g_layers.append(nn.Conv1d(last_layer_size, s, kernel_size=1, stride=1, groups=1))
            last_layer_size = s

        self.fully_layers = nn.ModuleList()

        assert last_layer_size % self.n_output_gaussians == 0
        last_layer_size = last_layer_size // self.n_output_gaussians + 1
        for s in fully_layer_sizes:
            self.fully_layers.append(nn.Conv1d(last_layer_size, s, kernel_size=1, stride=1, groups=1))
            last_layer_size = s

        self.output_layer = nn.Conv1d(last_layer_size, n_outputs_per_gaussian, kernel_size=1, stride=1, groups=1)

        self.image_output = True if output_image_width * output_image_height > 0 else False
        self.output_image_width = output_image_width
        self.output_image_height = output_image_height
        self.image_layers = nn.ModuleList()
        if self.image_output:
            self.image_layers.append(nn.Linear(self.per_g_layers[-1].out_channels + 1, output_image_width*output_image_height))

        self.name = "fit_gm_net_"
        self.name += name + "_g"
        for s in g_layer_sizes:
            self.name += f"_{s}"
        self.name += "__f"
        for s in fully_layer_sizes:
            self.name += f"_{s}"

        self.storage_path = "/home/madam/temp/prototype/" + self.name

    def save(self):
        print(f"gm_fitting.Net: saving to {self.storage_path}")
        torch.save(self.state_dict(), self.storage_path)

    def load(self):
        print(f"gm_fitting.Net: trying to load {self.storage_path}")
        if pathlib.Path(self.storage_path).is_file():
            state_dict = torch.load(self.storage_path)
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            # assert len(missing_keys) == 0
            # assert len(unexpected_keys) == 0
            print(f"gm_fitting.Net: loaded (missing: {missing_keys}, unexpected: {unexpected_keys}")
        else:
            print("gm_fitting.Net: not found")

    def device(self):
        return self.output_layer.bias.device

    def forward(self, data_in: MixtureReLUandBias, learning: bool = True) -> Mixture:
        n_layers = data_in.mixture.n_layers()
        n_input_components = data_in.mixture.n_components()
        n_dims = data_in.mixture.n_dimensions()

        data_normalised, normalisation_factors = gm.normalise(data_in)

        cov_data = data_normalised.mixture.covariances
        x = torch.cat((data_normalised.mixture.weights.view(n_layers, n_input_components, 1),
                       data_normalised.mixture.positions,
                       cov_data.view(n_layers, n_input_components, n_dims * n_dims)), dim=2)
        x = x.transpose(1, 2)

        for layer in self.per_g_layers:
            x = layer(x)
            x = F.relu(x)

        x = torch.sum(x, dim=2)

        i = x.view(n_layers, -1)
        i = torch.cat((data_normalised.bias.view(n_layers, 1), i), dim=1)
        for layer in self.image_layers:
            # for parameter in layer.parameters():
            #     print(f"parameter: {parameter.shape}")
            i = layer(i)
            i = F.leaky_relu(i)
        image = i.view(n_layers, self.output_image_width, self.output_image_height)


        # x is batch size x final g layer size now
        x = x.view(n_layers, -1, self.n_output_gaussians)
        x = torch.cat((data_normalised.bias.view(n_layers, 1, 1).expand(n_layers, 1, self.n_output_gaussians), x), dim=1)

        i = 0
        for layer in self.fully_layers:
            x = layer(x)
            x = F.relu(x)
            # x = self.batch_norms[i](x.view(-1, 1))
            i += 1

        x = self.output_layer(x)
        x = x.transpose(1, 2)

        # todo: those magic constants take care of scaling (important for the start). think of something generic, normalisation layer? input normalisation?
        weights = x[:, :, 0]
        positions = x[:, :, 1:(self.n_dims + 1)]
        # we are learning A, so that C = A @ A.T() + 0.01 * identity() is the resulting cov matrix
        A = x[:, :, (self.n_dims + 1):].view(n_layers, -1, self.n_dims, self.n_dims)
        C = A @ A.transpose(2, 3) + torch.eye(self.n_dims, self.n_dims, dtype=torch.float32, device=self.device()) * COVARIANCE_MIN
        covariances = C

        normalised_out = gm.Mixture(weights, positions, covariances)
        return gm.de_normalise(normalised_out, normalisation_factors), image


class Trainer:
    def __init__(self, net: Net, n_training_samples: int = 50 * 50, learning_rate: float = 0.001, save_weights: bool = False, testing_mode: bool = False):
        self.net = net
        self.n_training_samples = n_training_samples
        self.learning_rate = learning_rate
        self.save_weights = save_weights
        self.testing_mode = testing_mode
        self.criterion = nn.MSELoss()
        self.optimiser = optim.Adam(net.parameters(), lr=learning_rate)

    def train_on(self, data_in: gm.MixtureReLUandBias, epoch: int):
        data_in = data_in.detach()
        data_in, _ = gm.normalise(data_in)  # we normalise twice, but that shouldn't hurt (but performance). normalisation here is needed due to regularisation
        batch_size = data_in.mixture.n_layers()
        batch_start_time = time.perf_counter()
        self.optimiser.zero_grad()

        sampling_positions = torch.rand((batch_size, self.n_training_samples, data_in.mixture.n_dimensions()), dtype=torch.float32, device=self.net.device()) * 3 - 1.5
        target_sampling_values = data_in.evaluate_few_xes(sampling_positions)

        network_start_time = time.perf_counter()
        output_gm, image = self.net(data_in)
        network_time = time.perf_counter() - network_start_time

        eval_start_time = time.perf_counter()
        output_gm_sampling_values = output_gm.evaluate_few_xes(sampling_positions)
        criterion = self.criterion(output_gm_sampling_values, target_sampling_values)

        xv, yv = torch.meshgrid([torch.arange(-1.0, 1.0, 2 / 128, dtype=torch.float, device=data_in.device()),
                                 torch.arange(-1.0, 1.0, 2 / 128, dtype=torch.float, device=data_in.device())])
        xes = torch.cat((xv.reshape(-1, 1), yv.reshape(-1, 1)), 1).view(1, -1, 2).expand(batch_size, -1, 2)
        image_target = data_in.evaluate_few_xes(xes).view(-1, 128, 128)
        image_criterion = nn.MSELoss()(image, image_target)

        # the network was moving gaussians out of the sampling radius
        p = (output_gm.positions.abs() - 1)
        p = p.where(p > torch.zeros(1, device=p.device), torch.zeros(1, device=p.device))
        regularisation = p.sum()

        eval_time = time.perf_counter() - eval_start_time

        backward_start_time = time.perf_counter()
        loss = criterion + image_criterion + regularisation
        loss.backward()
        backward_time = time.perf_counter() - backward_start_time

        if self.testing_mode:
            for j in range(batch_size):
                data_in.mixture.debug_show(j, -2, -2, 2, 2, 0.05)
                data_in.debug_show(j, -2, -2, 2, 2, 0.05)
                i = image[j, :, :].detach().t().cpu()
                e = torch.zeros(256, 256)
                e[64:192, 64:192] = i
                plt.imshow(e, origin='lower')
                plt.show()
                output_gm.debug_show(j, -2, -2, 2, 2, 0.05)
                input("gm_fitting.Trainer: Press enter to continue")

        self.optimiser.step()

        info = (f"gm_fitting.Trainer: epoch = {epoch}:"
                f"batch loss {loss.item():.4f} (crit: {criterion.item()}, img_crit: {image_criterion.item()}, reg: {regularisation.item()}), "
                f"batch time = {time.perf_counter() - batch_start_time :.2f}s, "
                f"size = {batch_size}, "
                f"(forward: {network_time :.2f}s (per layer: {network_time / batch_size :.4f}s), eval: {eval_time :.3f}s, backward: {backward_time :.4f}s) ")
        print(info)
        if self.save_weights:
            self.net.save()
            f = open("/home/madam/temp/prototype/" + self.net.name + "_loss", "w")
            f.write(info)
            f.close()
