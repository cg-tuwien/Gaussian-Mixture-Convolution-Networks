import math
import time
import datetime
import pathlib
import typing
import numpy as np

import torch
import torch.nn.modules
import torch.utils.checkpoint
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard

from torch import Tensor

import madam_imagetools
import gm
import gm_fitting
import mat_tools


class GmConvolution(torch.nn.modules.Module):
    def __init__(self, n_layers_in: int, n_layers_out: int, n_kernel_components: int = 4, n_dims: int = 2,
                 position_range: float = 1, covariance_range: float = 0.25, weight_sd=0.1, weight_mean=0.0,
                 learn_positions: bool = True, learn_covariances: bool = True,
                 covariance_epsilon: float = 0.0001):
        super(GmConvolution, self).__init__()
        self.n_layers_in = n_layers_in
        self.n_layers_out = n_layers_out
        self.n_kernel_components = n_kernel_components
        self.n_dims = n_dims
        self.position_range = position_range
        self.covariance_range = covariance_range
        self.weight_sd = weight_sd
        self.covariance_epsilon = covariance_epsilon
        self.learn_positions = learn_positions
        self.learn_covariances = learn_covariances

        self.weights = torch.nn.modules.ParameterList()
        self.positions = torch.nn.modules.ParameterList()
        self.covariance_factors = torch.nn.modules.ParameterList()


        # todo: probably can optimise performance by putting kernels into their own dimension
        for i in range(self.n_layers_out):
            # positive mean produces a rather positive gm. i believe this is a better init
            weights = torch.randn(1, n_layers_in, n_kernel_components, 1, dtype=torch.float32) * weight_sd + weight_mean

            self.weights.append(torch.nn.Parameter(weights))
            if self.learn_positions:
                positions = torch.rand(1, n_layers_in, n_kernel_components, n_dims, dtype=torch.float32) * 2 * position_range - position_range
            else:
                assert(self.n_dims == 2)
                angles = torch.arange(0, 2 * math.pi, 2 * math.pi / (n_kernel_components - 1))
                xes = torch.cat((torch.zeros(1, dtype=torch.float), torch.sin(angles)), dim=0)
                yes = torch.cat((torch.zeros(1, dtype=torch.float), torch.cos(angles)), dim=0)
                positions = torch.cat((xes.view(-1, 1), yes.view(-1, 1)), dim=1) * position_range
                positions = positions.view(1, 1, n_kernel_components, 2).repeat((1, n_layers_in, 1, 1))
            self.positions.append(torch.nn.Parameter(positions))

            # initialise with a rather round covariance matrix
            # a psd matrix can be generated with A A'. we learn A and generate a pd matrix via  A A' + eye * epsilon
            covariance_factors = torch.rand(1, n_layers_in, n_kernel_components, n_dims, n_dims, dtype=torch.float32) * 2 - 1
            cov_rand_factor = 0.1 if self.learn_covariances else 0.0
            covariance_factors = covariance_factors * cov_rand_factor + torch.eye(self.n_dims)
            covariance_factors = covariance_factors * math.sqrt(covariance_range)
            self.covariance_factors.append(torch.nn.Parameter(covariance_factors))

        if not self.learn_positions:
            for t in self.positions:
                t.requires_grad = False

        if not self.learn_covariances:
            for t in self.covariance_factors:
                t.requires_grad = False

    def kernel(self, index: int):
        # a psd matrix can be generated with A A'. we learn A and generate a pd matrix via  A A' + eye * epsilon
        A = self.covariance_factors[index]
        covariances = A @ A.transpose(-1, -2) + torch.eye(self.n_dims, dtype=torch.float32, device=A.device) * self.covariance_epsilon
        kernel = torch.cat((self.weights[index], self.positions[index], covariances.view(1, self.n_layers_in, self.n_kernel_components, self.n_dims * self.n_dims)), dim=-1)
        assert gm.is_valid_mixture(kernel)
        return kernel

    def debug_render(self, position_range: float = None, image_size: int = 100, clamp: typing.Tuple[float, float] = (-0.3, 0.3)):
        if position_range is None:
            position_range = self.position_range * 2

        images = list()
        for i in range(self.n_layers_out):
            kernel = self.kernel(i)
            assert kernel.shape[0] == 1
            kernel_rendering = gm.render(kernel, x_low=-position_range, x_high=position_range, y_low=-position_range, y_high=position_range, width=image_size, height=image_size)
            images.append(kernel_rendering)
        images = torch.cat(images, dim=1)
        images = madam_imagetools.colour_mapped(images.cpu().numpy(), clamp[0], clamp[1])
        return images[:, :, :3]

    def forward(self, x: Tensor) -> Tensor:
        out_mixtures = []
        out_mixture_shape = list(x.shape)
        out_mixture_shape[1] = 1
        out_mixture_shape[2] = -1

        for i in range(self.n_layers_out):
            kernel = self.kernel(i)
            m = gm.convolve(x, kernel)
            out_mixtures.append(m.view(out_mixture_shape))

        return torch.cat(out_mixtures, dim=1)


class _NetCheckpointWrapper:
    def __init__(self, net, x, bias):
        self.net = net
        self.x = x
        self.bias = bias

    def __call__(self, *args, **kwargs):
        return self.net(self.x, self.bias)


class GmBiasAndRelu(torch.nn.modules.Module):
    def __init__(self, n_layers: int, n_output_gaussians: int = 10, n_dimensions=2, max_bias: float = 0.1):
        # todo support variable number of outputs and configurable net archs. needs a better init + training routine (start with few gaussians etc?)
        # assert n_output_gaussians == 10
        super(GmBiasAndRelu, self).__init__()
        self.n_layers = n_layers
        self.n_output_gaussians = n_output_gaussians
        # use a small bias for the start. i hope it's easier for the net to increase it than to lower it
        self.bias = torch.nn.Parameter(torch.rand(1, self.n_layers) * max_bias)

        self.net = gm_fitting.Net([64, 128, 256, 512, 512, n_output_gaussians * 25],
                                  [256, 256, 256, 256, 256, 128],
                                  n_output_gaussians=n_output_gaussians,
                                  n_dims=n_dimensions,
                                  n_agrs=1, batch_norm=True)
        self.net.load(strict=False)
        # if not self.net.load(strict=True):
        #     raise Exception(f"Fitting network {self.net.name} not found.")
        self.train_fitting_flag = False
        self.train_fitting(self.train_fitting_flag)

        self.name = f"GmBiasAndRelu_{n_layers}_{n_output_gaussians}"
        self.storage_path = self.net.storage_path

        print(self.net)
        # todo: option to make fitting net have common or seperate weights per module

    def train_fitting(self, flag: bool):
        self.train_fitting_flag = flag
        if flag:
            self.net.requires_grad_(True)
            self.bias.requires_grad_(False)
        else:
            self.net.requires_grad_(False)
            self.bias.requires_grad_(True)

    def forward(self, x: Tensor, overwrite_bias: Tensor = None, division_axis: int = 0) -> Tensor:
        # todo: think of something that would make it possible to do live learning of the fitting network
        n_dimensions = gm.n_dimensions(x)
        n_components = gm.n_components(x)


        if n_components < 134 or True:
            bias = self.bias if overwrite_bias is None else overwrite_bias
            bias = torch.abs(bias)
            return self.net(x, bias)[0]
            # if self.train_fitting_flag:
            #     wrapper = _NetCheckpointWrapper(self.net, x, bias)
            #     net_params = tuple(self.net.parameters())
            #     result = torch.utils.checkpoint.checkpoint(wrapper, *net_params)[0]
            #
            # else:
            #     result = torch.utils.checkpoint.checkpoint(self.net, x, bias)[0]
            # return result
        else:
            sorted_indices = torch.argsort(gm.positions(x.detach())[:, :, :, division_axis])
            sorted_mixture = mat_tools.my_index_select(x, sorted_indices)

            division_index = n_components // 2
            next_division_axis = (division_axis + 1) % n_dimensions

            fitted_left = self.forward(sorted_mixture[:, :, :division_index], overwrite_bias=overwrite_bias, division_axis=next_division_axis)
            fitted_right = self.forward(sorted_mixture[:, :, division_index:], overwrite_bias=overwrite_bias, division_axis=next_division_axis)

            return torch.cat((fitted_left, fitted_right), dim=2)

    def save(self):
        self.net.save()


class BatchNorm(torch.nn.modules.Module):
    def __init__(self, per_gaussian_norm: bool = False):
        super(BatchNorm, self).__init__()
        self.per_gaussian_norm = per_gaussian_norm

    def forward(self, x: Tensor) -> Tensor:
        integral = gm.integrate(x).view(gm.n_batch(x), gm.n_layers(x), 1)
        if not self.per_gaussian_norm:
            integral = torch.mean(integral, dim=0, keepdim=True)

        weights = gm.weights(x)
        positions = gm.positions(x)
        covariances = gm.covariances(x)

        weights = weights / integral
        return gm.pack_mixture(weights, positions, covariances)


class MaxPooling(torch.nn.modules.Module):
    def __init__(self, n_output_gaussians: int = 10):
        super(MaxPooling, self).__init__()
        self.n_output_gaussians = n_output_gaussians

    def forward(self, x: Tensor) -> Tensor:
        sorted_indices = torch.argsort(gm.integrate_components(x.detach()), dim=2, descending=True)
        sorted_mixture = mat_tools.my_index_select(x, sorted_indices)

        n_output_gaussians = min(self.n_output_gaussians, gm.n_components(x))
        return sorted_mixture[:, :, :n_output_gaussians, :]
