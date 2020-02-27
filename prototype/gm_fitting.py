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
from torch import Tensor

import gm
import config
import madam_imagetools
import mat_tools

COVARIANCE_MIN = 0.0001


class Net(nn.Module):
    def __init__(self, class_name: str, config_name: str, n_dims: int):
        super(Net, self).__init__()
        self.class_name = class_name
        self.config_name = config_name
        self.name = "gmfit_" + self.class_name + "_" + self.config_name
        self.n_dims = n_dims
        self.storage_path = config.data_base_path / "weights" / self.name

    def save(self, storage_path: str = None) -> None:
        if storage_path is None:
            storage_path = self.storage_path
        print(f"gm_fitting.{self.class_name}: saving to {storage_path}")
        torch.save(self.state_dict(), storage_path)

    def load(self, storage_path: str = None, strict: bool = False) -> bool:
        if storage_path is None:
            storage_path = self.storage_path

        print(f"gm_fitting.{self.class_name}: trying to load {storage_path}")
        if pathlib.Path(storage_path).is_file():
            state_dict = torch.load(storage_path, map_location=torch.device('cpu'))
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)
            # assert len(missing_keys) == 0
            # assert len(unexpected_keys) == 0
            print(f"gm_fitting.{self.class_name}: loaded (missing: {missing_keys}, unexpected: {unexpected_keys}")
            return True
        else:
            print(f"gm_fitting.{self.class_name}: not found")
            return False

    def device(self) -> torch.device:
        return next(self.parameters()).device


class PointNetToLatent(nn.Module):
    def __init__(self, g_layer_sizes: typing.List, n_dims: int = 2, batch_norm: bool = True, aggregations: int = 1):
        super(PointNetToLatent, self).__init__()
        self.g_layer_sizes = g_layer_sizes
        self.batch_norm = batch_norm
        self.aggregations = aggregations

        n_inputs_per_gaussian = 1 + n_dims + n_dims * n_dims
        last_layer_size = n_inputs_per_gaussian
        module_list = nn.ModuleList()
        for s in g_layer_sizes[:-1]:
            module_list.append(nn.Conv1d(last_layer_size, s, kernel_size=1, stride=1, groups=1))
            if batch_norm:
                module_list.append(nn.BatchNorm1d(s))
            module_list.append(torch.nn.LeakyReLU())
            last_layer_size = s
        module_list.append(nn.Conv1d(last_layer_size, g_layer_sizes[-1], kernel_size=1, stride=1, groups=1))

        self.net = torch.nn.Sequential(*module_list)
        self.latent_layer_size = g_layer_sizes[-1]

    def forward(self, mixture_in: Tensor) -> Tensor:
        n_batch = gm.n_batch(mixture_in)
        n_layers = gm.n_layers(mixture_in)
        n_input_components = gm.n_components(mixture_in)

        x = mixture_in.view(n_batch * n_layers, n_input_components, -1)

        # component should be the last dimension for conv1d to work
        x = x.transpose(1, 2)

        # this goes through all the layers, the rest of the code is just massaging
        x = self.net(x)

        aggregations = self.aggregations
        aggregations_list = []
        n_out = x.shape[1]
        x_sum = torch.sum(x[:, 0:(n_out // aggregations) * 1, :], dim=2).view(n_batch * n_layers, -1, 1)
        aggregations_list.append(x_sum)
        if aggregations >= 4:
            x_max = torch.max(x[:, (n_out // aggregations) * 3:(n_out // aggregations) * 4, :], dim=2).values.view(n_batch * n_layers, -1, 1)
            aggregations_list.append(x_max)
        if aggregations >= 3:
            x_var = torch.var(x[:, (n_out // aggregations) * 2:(n_out // aggregations) * 3, :], dim=2, unbiased=False).view(n_batch * n_layers, -1, 1)
            aggregations_list.append(x_var)
        if aggregations >= 2:
            x_prd = torch.abs(x[:, (n_out // aggregations) * 1:(n_out // aggregations) * 2, :] + 1)
            # old, explods when switchng from n_gaussians=10 to 100
            # x_prd = torch.prod(x_prd, dim=2).view(n_layers, -1, 1)

            x_prd = torch.max(x_prd, torch.tensor(0.01, dtype=torch.float32, device=x.device))
            x_prd = torch.mean(torch.log(x_prd), dim=2).view(n_batch * n_layers, -1, 1)
            x_prd = torch.min(x_prd, torch.tensor(10, dtype=torch.float32, device=x.device))
            x_prd = torch.exp(x_prd)
            aggregations_list.append(x_prd)

        x = torch.cat(aggregations_list, dim=2).reshape(n_batch, n_layers, -1)
        return x

    def config_name(self) -> str:
        config_name = "_bn" if self.batch_norm else ''
        config_name += f"_a{self.aggregations}" if self.aggregations > 1 else ''
        config_name += "_g"
        for s in self.g_layer_sizes:
            config_name += f"_{s}"
        return config_name


def raw_out_to_gm(x: Tensor) -> Tensor:
    n_batch = gm.n_batch(x)
    n_layers = gm.n_layers(x)
    n_dims = gm.n_dimensions(x)
    n_components = gm.n_components(x)

    weights = x[:, :, :, 0]
    positions = x[:, :, :, 1:(n_dims + 1)]
    # we are learning A, so that C = A @ A.T() + 0.01 * identity() is the resulting cov matrix
    A = x[:, :, :, (n_dims + 1):].view(n_batch, n_layers, n_components, n_dims, n_dims)
    C = A @ A.transpose(-2, -1) + torch.eye(n_dims, n_dims, dtype=torch.float32, device=x.device) * COVARIANCE_MIN
    covariances = C

    m = gm.pack_mixture(weights.view(n_batch, n_layers, n_components),
                        positions.view(n_batch, n_layers, n_components, n_dims),
                        covariances.view(n_batch, n_layers, n_components, n_dims, n_dims))
    assert gm.is_valid_mixture(m)
    return m


class PointNetWithMLP(Net):
    def __init__(self,
                 g_layer_sizes: typing.List,
                 fully_layer_sizes: typing.List,
                 n_output_gaussians: int,
                 name: str = "",
                 n_dims: int = 2,
                 aggregations: int = 4,
                 batch_norm: bool = False):
        config_name = name
        point_net_to_latent_module = PointNetToLatent(g_layer_sizes, n_dims=n_dims, batch_norm=batch_norm, aggregations=aggregations)
        config_name += point_net_to_latent_module.config_name()
        config_name += "_f"
        for s in fully_layer_sizes:
            config_name += f"_{s}"

        super(PointNetWithMLP, self).__init__("PointNetWithMLP", config_name, n_dims)

        self.point_net_to_latent_module = point_net_to_latent_module

        self.n_output_gaussians = n_output_gaussians
        n_outputs_per_gaussian = 1 + n_dims + n_dims * n_dims

        fully_layers = nn.ModuleList()

        assert point_net_to_latent_module.latent_layer_size % (self.n_output_gaussians * aggregations) == 0
        last_layer_size = point_net_to_latent_module.latent_layer_size + 1
        for s in fully_layer_sizes:
            fully_layers.append(nn.Linear(last_layer_size, s))
            if batch_norm:
                fully_layers.append(nn.BatchNorm1d(s))
            fully_layers.append(nn.LeakyReLU())
            last_layer_size = s

        fully_layers.append(nn.Linear(last_layer_size, n_outputs_per_gaussian * self.n_output_gaussians))

        self.latent_to_raw_out = nn.Sequential(*fully_layers)

    def forward(self, mixture_in: Tensor, bias_in: Tensor, latent_space_vectors: typing.List[Tensor] = None) -> Tensor:
        n_batch = gm.n_batch(mixture_in)
        n_layers = gm.n_layers(mixture_in)

        mixtures_normalised, bias_normalised, normalisation_factors = gm.normalise(mixture_in, bias_in)

        x = self.point_net_to_latent_module(mixtures_normalised).view(n_batch * n_layers, -1)
        if latent_space_vectors is not None:
            latent_space_vectors.append(x.detach())

        bias_extended = bias_normalised.view(-1, n_layers, 1).expand(n_batch, n_layers, 1).view(n_batch * n_layers, 1)
        x = torch.cat((bias_extended, x), dim=1)

        # this goes through all the output layers, the rest of the code is just massaging
        x = self.latent_to_raw_out(x)

        x = x.view(n_batch, n_layers, self.n_output_gaussians, -1)
        x = raw_out_to_gm(x)

        return gm.de_normalise(x, normalisation_factors)


class PointNetWithParallelMLPs(Net):
    def __init__(self,
                 g_layer_sizes: typing.List,
                 fully_layer_sizes: typing.List,
                 n_output_gaussians: int,
                 name: str = "",
                 n_dims: int = 2,
                 aggregations: int = 4,
                 batch_norm: bool = False):
        config_name = name
        point_net_to_latent_module = PointNetToLatent(g_layer_sizes, n_dims=n_dims, batch_norm=batch_norm, aggregations=aggregations)
        config_name += point_net_to_latent_module.config_name()
        config_name += "_f"
        for s in fully_layer_sizes:
            config_name += f"_{s}"

        super(PointNetWithParallelMLPs, self).__init__("PointNetWithParallelMLPs", config_name, n_dims)

        self.point_net_to_latent_module = point_net_to_latent_module

        self.n_output_gaussians = n_output_gaussians
        n_outputs_per_gaussian = 1 + n_dims + n_dims * n_dims

        fully_layers = nn.ModuleList()

        assert point_net_to_latent_module.latent_layer_size % (self.n_output_gaussians * aggregations) == 0
        last_layer_size = point_net_to_latent_module.latent_layer_size // self.n_output_gaussians + 1
        for s in fully_layer_sizes:
            fully_layers.append(nn.Conv1d(last_layer_size, s, kernel_size=1, stride=1, groups=1))
            if batch_norm:
                fully_layers.append(nn.BatchNorm1d(s))
            fully_layers.append(nn.LeakyReLU())
            last_layer_size = s

        fully_layers.append(nn.Conv1d(last_layer_size, n_outputs_per_gaussian, kernel_size=1, stride=1, groups=1))

        self.latent_to_raw_out = nn.Sequential(*fully_layers)

    def forward(self, mixture_in: Tensor, bias_in: Tensor, learning: bool = True, latent_space_vectors: typing.List[Tensor] = None) -> Tensor:
        n_batch = gm.n_batch(mixture_in)
        n_layers = gm.n_layers(mixture_in)

        mixtures_normalised, bias_normalised, normalisation_factors = gm.normalise(mixture_in, bias_in)

        x = self.point_net_to_latent_module(mixtures_normalised).view(n_batch * n_layers, -1)
        if latent_space_vectors is not None:
            latent_space_vectors.append(x.detach())

        # x is batch size x final g layer size now
        x = x.view(n_batch * n_layers, -1, self.n_output_gaussians)
        bias_extended = bias_normalised.view(-1, n_layers, 1, 1).expand(n_batch, n_layers, 1, self.n_output_gaussians).view(n_batch * n_layers, 1, self.n_output_gaussians)
        x = torch.cat((bias_extended, x), dim=1)

        # this goes through all the output layers, the rest of the code is just massaging
        x = self.latent_to_raw_out(x)

        x = x.view(n_batch, n_layers, self.n_output_gaussians, -1)
        x = raw_out_to_gm(x)

        return gm.de_normalise(x, normalisation_factors)


class SpaceSubdivider(Net):
    def __init__(self, generate_fitting_module: typing.Callable, n_input_gaussians: int, n_output_gaussians: int):
        net: Net = generate_fitting_module(n_input_gaussians, 10)
        config_name = net.name
        super(PointNetWithParallelMLPs, self).__init__("SpaceSubdivider", config_name, net.n_dims)

        self.n_output_gaussians = n_output_gaussians

        self.net = net
        self.net.load(strict=False)
        # if not self.net.load(strict=True):
        #     raise Exception(f"Fitting network {self.net.name} not found.")

    def forward(self, x: Tensor, overwrite_bias: Tensor = None, division_axis: int = 0, latent_space_vectors: typing.List[Tensor] = None) -> Tensor:
        n_dimensions = gm.n_dimensions(x)
        n_components = gm.n_components(x)

        if n_components < 134:
            bias = self.bias if overwrite_bias is None else overwrite_bias
            bias = torch.abs(bias)
            result = self.net(x, bias, latent_space_vectors=latent_space_vectors)
            # if self.train_fitting_flag:
            #     wrapper = _NetCheckpointWrapper(self.net, x, bias)
            #     net_params = tuple(self.net.parameters())
            #     result = torch.utils.checkpoint.checkpoint(wrapper, *net_params)[0]
            #
            # else:
            #     result = torch.utils.checkpoint.checkpoint(self.net, x, bias)[0]
        else:
            sorted_indices = torch.argsort(gm.positions(x.detach())[:, :, :, division_axis])
            sorted_mixture = mat_tools.my_index_select(x, sorted_indices)

            division_index = n_components // 2
            next_division_axis = (division_axis + 1) % n_dimensions

            # todo concatenate latent space vecotr if is not null:: that is actually easy, but we the resorting will foo things up. do we need a safe guard? the result would be only usefull for drawing latent space
            fitted_left = self.forward(sorted_mixture[:, :, :division_index], overwrite_bias=overwrite_bias, division_axis=next_division_axis)
            fitted_right = self.forward(sorted_mixture[:, :, division_index:], overwrite_bias=overwrite_bias, division_axis=next_division_axis)

            result = torch.cat((fitted_left, fitted_right), dim=2)

        return result


class Sampler:
    def __init__(self, net: Net, n_training_samples: int = 50 * 50, learning_rate: float = 0.001):
        self.net = net
        self.n_training_samples = n_training_samples
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()
        self.optimiser = optim.Adam(net.parameters(), lr=learning_rate)
        self.tensor_board_graph_written = False
        self.storage_path = self.net.storage_path.with_suffix(".optimiser_state")

    @staticmethod
    def log_image(tensor_board_writer: torch.utils.tensorboard.SummaryWriter, tag: str, image: typing.Sequence[torch.Tensor], epoch: int, clamp: typing.Sequence[float]):
        image = image.detach().t().cpu().numpy()
        image = madam_imagetools.colour_mapped(image, clamp[0], clamp[1])
        tensor_board_writer.add_image(tag, image, epoch, dataformats='HWC')

    @staticmethod
    def log_images(tensor_board_writer: torch.utils.tensorboard.SummaryWriter, tag: str, images: typing.Sequence[torch.Tensor], epoch: int, clamp: typing.Sequence[float]):
        for i in range(len(images)):
            images[i] = images[i].detach().t().cpu().numpy()
            images[i] = madam_imagetools.colour_mapped(images[i], clamp[0], clamp[1])
            images[i] = images[i][:, :, 0:3]
            images[i] = images[i].reshape(1, images[i].shape[0], images[i].shape[1], 3)
        images = np.concatenate(images, axis=0)
        tensor_board_writer.add_image(tag, images, epoch, dataformats='NHWC')

    def run_on(self, mixture_in: Tensor, bias_in: Tensor, epoch: int = None, train: bool = True, tensor_board_writer: torch.utils.tensorboard.SummaryWriter = None) -> float:
        mixture_in = mixture_in.detach()
        bias_in = bias_in.detach()
        assert gm.is_valid_mixture_and_bias(mixture_in, bias_in)

        mixture_in, bias_in, _ = gm.normalise(mixture_in, bias_in)  # we normalise twice, but that shouldn't hurt (but performance). normalisation here is needed due to regularisation
        # gm.debug_show_with_activation_fun(mixture_in, bias_in, batch_i=0, layer_i=0, x_low=-1.2, y_low=-1.2, x_high=1.2, y_high=1.2, step=0.02)
        batch_size = gm.n_batch(mixture_in)
        n_dims = gm.n_dimensions(mixture_in)
        batch_start_time = time.perf_counter()
        self.optimiser.zero_grad()

        sampling_positions = torch.rand((1, 1, self.n_training_samples, n_dims), dtype=torch.float32, device=mixture_in.device) * 3 - 1.5
        target_sampling_values = gm.evaluate_with_activation_fun(mixture_in, bias_in, sampling_positions)

        network_start_time = time.perf_counter()
        net_result = self.net(mixture_in, bias_in)
        network_time = time.perf_counter() - network_start_time

        if isinstance(net_result, tuple):
            output_gm, latent_vector = net_result
        else:
            output_gm = net_result
            latent_vector = torch.zeros(1, 1)

        eval_start_time = time.perf_counter()
        output_gm_sampling_values = gm.evaluate(output_gm, sampling_positions)
        criterion = self.criterion(output_gm_sampling_values, target_sampling_values) * 2

        # the network was moving gaussians out of the sampling radius
        p = (gm.positions(output_gm).abs() - 1)
        p = p.where(p > torch.zeros(1, device=p.device), torch.zeros(1, device=p.device))
        regularisation = p.mean()

        eval_time = time.perf_counter() - eval_start_time

        backward_start_time = time.perf_counter()
        loss = criterion + regularisation

        backward_time = 0
        if train:
            loss.backward()
            backward_time = time.perf_counter() - backward_start_time

            self.optimiser.step()

        if tensor_board_writer is not None:
            # if not self.tensor_board_graph_written:
            #     tensor_board_graph_written = True
            #     tensor_board_writer.add_graph(self.net, data_in)
            tensor_board_writer.add_scalar(f"{self.net.name}_fitting 0. batch_loss", loss.item(), epoch)
            # tensor_board_writer.add_scalar(f"{self.net.name}_fitting 1. criterion", criterion.item(), epoch)
            # tensor_board_writer.add_scalar(f"fitting 3. regularisation_aggr_prod", regularisation_aggr_prod.item(), epoch)
            tensor_board_writer.add_scalar(f"{self.net.name}_fitting 4. whole time", time.perf_counter() - batch_start_time, epoch)
            # tensor_board_writer.add_scalar(f"{self.net.name}_fitting 5. network_time", network_time, epoch)
            tensor_board_writer.add_scalar(f"{self.net.name}_fitting 6. eval_time", eval_time, epoch)
            # tensor_board_writer.add_scalar(f"{self.net.name}_fitting 7. backward_time", backward_time, epoch)

            if epoch is None or (epoch % 10 == 0 and epoch < 100) or (epoch % 100 == 0 and epoch < 1000) or (epoch % 1000 == 0 and epoch < 10000) or (epoch % 10000 == 0):
                image_size = 80
                xv, yv = torch.meshgrid([torch.arange(-1.2, 1.2, 2.4 / image_size, dtype=torch.float, device=mixture_in.device),
                                         torch.arange(-1.2, 1.2, 2.4 / image_size, dtype=torch.float, device=mixture_in.device)])
                xes = torch.cat((xv.reshape(-1, 1), yv.reshape(-1, 1)), 1).view(1, 1, -1, 2)

                n_shown_images = 10
                n_batches_eval = n_shown_images // gm.n_layers(mixture_in) + 1
                n_batches_eval = min(n_batches_eval, gm.n_batch(mixture_in))
                n_layers_eval = min(gm.n_layers(mixture_in), n_shown_images)
                mixture_eval = mixture_in.detach()[:n_batches_eval, :n_layers_eval, :, :]
                bias_eval = bias_in.detach()[:n_batches_eval, :n_layers_eval]

                image_target = gm.evaluate_with_activation_fun(mixture_eval, bias_eval, xes).view(-1, image_size, image_size)
                fitted_mixture_image = gm.evaluate(output_gm.detach(), xes).view(-1, image_size, image_size)
                self.log_images(tensor_board_writer,
                                f"{self.net.name}_fitting target_prediction",
                                [image_target[:n_shown_images, :, :].transpose(0, 1).reshape(image_size, -1),
                                 fitted_mixture_image[:n_shown_images, :, :].transpose(0, 1).reshape(image_size, -1)],
                                epoch, [-0.5, 2])
                # self.log_image("latent_space", latent_vector.detach(), epoch, (-5, 5))

        info = (f"gm_fitting.Trainer: epoch = {epoch}:"
                f"batch loss {loss.item():.4f} (crit: {criterion.item()} (rest is regularisation)), "
                f"batch time = {time.perf_counter() - batch_start_time :.2f}s, "
                f"size = {batch_size}, "
                f"(forward: {network_time :.2f}s (per layer: {network_time / batch_size :.4f}s), eval: {eval_time :.3f}s, backward: {backward_time :.4f}s) ")
        print(info)
        return criterion.item()

    def to_(self, device: torch.device):
        for state in self.optimiser.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    def load(self, strict: bool = False):
        print(f"gm_fitting.Trainer: trying to load {self.storage_path}")
        if pathlib.Path(self.storage_path).is_file():
            state_dict = torch.load(self.storage_path, map_location=torch.device('cpu'))
            self.optimiser.load_state_dict(state_dict)
            # assert len(missing_keys) == 0
            # assert len(unexpected_keys) == 0
            print(f"gm_fitting.Trainer: loaded")
            return True
        else:
            print("gm_fitting.Trainer: not found")
            assert not strict
            return False

    def save_optimiser_state(self):
        print(f"gm_fitting.Trainer: saving to {self.storage_path}")
        torch.save(self.optimiser.state_dict(), self.storage_path)
