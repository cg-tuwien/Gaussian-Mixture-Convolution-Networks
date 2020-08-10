import math
import time
import datetime
import pathlib
import typing
import numpy as np

import torch
import torch.nn.modules
import torch.nn.functional
import torch.utils.checkpoint
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard

from torch import Tensor

import gmc.mixture as gm
import gmc.mat_tools as mat_tools
import gmc.image_tools as madam_imagetools
import prototype_convolution.config as config
import prototype_convolution.fitting_em as fitting_em


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

        # todo: probably can optimise performance by putting kernels into their own dimension (currently as a list)
        for i in range(self.n_layers_out):
            # positive mean produces a rather positive gm. i believe this is a better init
            weights = torch.randn(1, n_layers_in, n_kernel_components, 1, dtype=torch.float32) * weight_sd + weight_mean
            self.weights.append(torch.nn.Parameter(weights))

            if self.learn_positions and False:
                positions = torch.rand(1, n_layers_in, n_kernel_components, n_dims, dtype=torch.float32) * 2 - 1
            else:
                assert (self.n_dims == 2)
                angles = torch.arange(0, 2 * math.pi, 2 * math.pi / (n_kernel_components - 1))
                xes = torch.cat((torch.zeros(1, dtype=torch.float), torch.sin(angles)), dim=0)
                yes = torch.cat((torch.zeros(1, dtype=torch.float), torch.cos(angles)), dim=0)
                positions = torch.cat((xes.view(-1, 1), yes.view(-1, 1)), dim=1)
                positions = positions.view(1, 1, n_kernel_components, 2).repeat((1, n_layers_in, 1, 1))
            self.positions.append(torch.nn.Parameter(positions))

            # initialise with a rather round covariance matrix
            # a psd matrix can be generated with A A'. we learn A and generate a pd matrix via  A A' + eye * epsilon
            covariance_factors = torch.rand(1, n_layers_in, n_kernel_components, n_dims, n_dims, dtype=torch.float32) * 2 - 1
            cov_rand_factor = 0
            covariance_factors = covariance_factors * cov_rand_factor + torch.eye(self.n_dims)
            covariance_factors = covariance_factors
            self.covariance_factors.append(torch.nn.Parameter(covariance_factors))
            assert(gm.is_valid_mixture(
                gm.pack_mixture(weights,
                                positions,
                                covariance_factors @ covariance_factors.transpose(-1, -2) + torch.eye(self.n_dims, dtype=torch.float32, device=covariance_factors.device) * self.covariance_epsilon)))

        self.set_requires_grad(True)

    def set_requires_grad(self, flag: bool):
        for t in self.weights:
            t.requires_grad = flag

        for t in self.positions:
            t.requires_grad = self.learn_positions and flag

        for t in self.covariance_factors:
            t.requires_grad = self.learn_covariances and flag

    def kernel(self, index: int):
        # a psd matrix can be generated with A A'. we learn A and generate a pd matrix via  A A' + eye * epsilon
        A = self.covariance_factors[index]
        covariances = A @ A.transpose(-1, -2) + torch.eye(self.n_dims, dtype=torch.float32, device=A.device) * self.covariance_epsilon
        # kernel = torch.cat((self.weights[index], self.positions[index], covariances.view(1, self.n_layers_in, self.n_kernel_components, self.n_dims * self.n_dims)), dim=-1)
        kernel = gm.pack_mixture(self.weights[index], self.positions[index] * self.position_range, covariances * self.covariance_range)
        assert gm.is_valid_mixture(kernel)
        return kernel

    def regularisation_loss(self):
        cost = torch.zeros(1, dtype=torch.float32, device=self.weights[0].device)
        for i in range(self.n_layers_out):
            A = self.covariance_factors[i]

            # problem: symeig produces NaN gradients when the eigenvalues are the same (symmetric gaussian).
            # this shouldn't be a problem because we mask them out anyways. however, pytorch doesn't differentiate between zero and no gradient. and so 0 * NaN = NaN
            # the only possibility to avoid this is masking the offending data before the NaN producing operation (and that's what we do, hence the 2 iterations)
            # tracked in here; https://github.com/pytorch/pytorch/issues/23156#issuecomment-528663523

            covariances = A @ A.transpose(-1, -2)
            # eigenvalues = torch.symeig(covariances, eigenvectors=True).eigenvalues.detach()
            # largest_eigenvalue = eigenvalues[:, :, :, -1]
            # smallest_eigenvalue = eigenvalues[:, :, :, 0]
            # cov_cost: Tensor = 0.1 * largest_eigenvalue / smallest_eigenvalue - 1
            # # t o d o: this will not work for 3d (round disk)
            # covariances = covariances.where(cov_cost.view(1, self.n_layers_in, self.n_kernel_components, 1, 1) > torch.zeros_like(cost),
            #                                 torch.tensor([[[[[2, 0], [0, 1]]]]], dtype=torch.float32, device=self.weights[0].device))
            # well, well. I can't provoke this problem. it should be there but it isn't. maybe they fixed it in the meanwhile without documenting.

            eigenvalues = torch.symeig(covariances, eigenvectors=True).eigenvalues
            largest_eigenvalue = eigenvalues[:, :, :, -1]
            smallest_eigenvalue = eigenvalues[:, :, :, 0]
            cov_cost: Tensor = 0.1 * largest_eigenvalue / smallest_eigenvalue - 1
            cov_cost = cov_cost.where(cov_cost > torch.zeros_like(cost), torch.zeros_like(cost))

            cov_cost2: Tensor = largest_eigenvalue - self.position_range
            cov_cost2 = cov_cost.where(cov_cost2 > torch.zeros_like(cost), torch.zeros_like(cost))

            positions = self.positions[i]
            distances = ((positions ** 2).sum(dim=-1) + self.position_range * 0.001).sqrt()
            distance_cost = distances - torch.ones_like(cost) * self.position_range
            distance_cost = distance_cost.where(distance_cost > torch.zeros_like(cost), torch.zeros_like(cost))

            cost = cost + cov_cost.sum() + cov_cost2.sum() + distance_cost.sum()

        return cost


    def debug_render(self, position_range: float = None, image_size: int = 80, clamp: typing.Tuple[float, float] = (-0.3, 0.3)):
        if position_range is None:
            position_range = self.position_range * 2

        images = list()
        for i in range(self.n_layers_out):
            kernel = self.kernel(i)
            assert kernel.shape[0] == 1
            kernel_rendering = gm.render(kernel, x_low=-position_range*1.25, x_high=position_range*1.25, y_low=-position_range*1.25, y_high=position_range*1.25, width=image_size, height=image_size)
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

#
# def generate_default_fitting_module(n_input_gaussians: int, n_output_gaussians: int) -> fitting_net.Net:
#     assert n_output_gaussians > 0
#     n_dimensions = 2
#     return fitting_net.PointNetWithParallelMLPs([64, 128, 256, 512, 512, n_output_gaussians * 25],
#                                                 [256, 256, 256, 256, 256, 128],
#                                                 n_output_gaussians=n_output_gaussians,
#                                                 n_dims=n_dimensions,
#                                                 aggregations=1, batch_norm=True)


class GmBiasAndRelu(torch.nn.modules.Module):
    def __init__(self, layer_id: str, n_layers: int, n_output_gaussians: int, n_input_gaussians: int = -1, use_bias: bool = False):
        # todo: option to make fitting net have common or seperate weights per module
        super(GmBiasAndRelu, self).__init__()
        self.layer_id = layer_id
        self.n_layers = n_layers
        self.n_input_gaussians = n_input_gaussians
        self.n_output_gaussians = n_output_gaussians

        # use a small bias for the start. i hope it's easier for the net to increase it than to lower it
        self.bias = torch.nn.Parameter(torch.zeros(1, self.n_layers) - 0.1)
        self.use_bias = use_bias

        # WARNING !!!: evil code. the string self.gm_fitting_net_666 is used for filtering in experiment_gm_mnist_model.Net.save_model(). !!! WARNING
        # todo: fix it
        # self.gm_fitting_net_666: fitting_net.Net = generate_fitting_module(n_input_gaussians, n_output_gaussians)

        # self.gm_fitting_net_666.requires_grad_(True)
        self.bias.requires_grad_(True)
        # self.train_fitting(False)

        # self.name = f"GmBiasAndRelu_{layer_id}"
        # self.storage_path = config.data_base_path / "weights" / f"GmBiasAndRelu_{layer_id}_{self.gm_fitting_net_666.name}"

        self.last_in = None
        self.last_out = None

        # self.fitting_sampler = fitting_net.Sampler(self, n_training_samples=1000)

        # print(self.gm_fitting_net_666)

    def train_fitting(self, flag: bool):
        # self.gm_fitting_net_666.requires_grad_(flag)
        self.bias.requires_grad_(not flag)

    def set_requires_grad(self, flag: bool):
        # self.gm_fitting_net_666.requires_grad_(flag)
        self.bias.requires_grad_(flag)

    def forward(self, x: Tensor, overwrite_bias: Tensor = None) -> Tensor:
        bias = self.bias if overwrite_bias is None else overwrite_bias

        if self.use_bias:
            # before there was a torch.abs(bias), but that is not differentiable at b == 0 + something weird is happening if bias < 0
            bias = torch.nn.functional.softplus(bias, beta=20)
        else:
            bias = torch.zeros_like(bias)

        y_m, y_bias = fitting_em.relu(x, bias)
        result = fitting_em.em_algorithm(y_m, n_fitting_components=self.n_output_gaussians)[0]

        self.last_in = x.detach()
        self.last_out = result.detach()
        return result

    def debug_render(self, position_range: typing.Tuple[float, float, float, float] = None, image_size: int = 80, clamp: typing.Tuple[float, float] = (-1.0, 1.0)):
        if position_range is None:
            position_range = [-1.0, -1.0, 1.0, 1.0]

        last_in = gm.render(self.last_in, batches=[0, 1], layers=[0, None],
                            x_low=position_range[0], y_low=position_range[1], x_high=position_range[2], y_high=position_range[3],
                            width=image_size, height=image_size)
        target = gm.render_bias_and_relu(self.last_in, torch.nn.functional.softplus(self.bias.detach(), beta=20), batches=[0, 1], layers=[0, None],
                                         x_low=position_range[0], y_low=position_range[1], x_high=position_range[2], y_high=position_range[3],
                                         width=image_size, height=image_size)
        prediction = gm.render(self.last_out, batches=[0, 1], layers=[0, None],
                               x_low=position_range[0], y_low=position_range[1], x_high=position_range[2], y_high=position_range[3],
                               width=image_size, height=image_size)
        images = [last_in, target, prediction]
        images = torch.cat(images, dim=1)
        images = madam_imagetools.colour_mapped(images.cpu().numpy(), clamp[0], clamp[1])
        return images[:, :, :3]

    def save_fitting_parameters(self):
        # todo: make nicer, we want facilities to separate the learned parameters from fitting and gaussian kernels / bias
        # print(f"gm_modules.GmBiasAndRelu: saving fitting module to {self.storage_path}")
        # self.gm_fitting_net_666.save(self.storage_path)
        pass

    def load_fitting_parameters(self, strict: bool = False) -> bool:
        # print(f"gm_modules.GmBiasAndRelu: trying to load fitting module from {self.storage_path}")
        # if not self.gm_fitting_net_666.load(self.storage_path, strict=strict):
        #     self.gm_fitting_net_666.load(strict=strict)
        pass


class BatchNorm(torch.nn.modules.Module):
    def __init__(self, per_mixture_norm: bool = False, per_layer_norm: bool = False):
        super(BatchNorm, self).__init__()
        self.per_mixture_norm = per_mixture_norm
        self.per_layer_norm = per_layer_norm

    def forward(self, x: Tensor) -> Tensor:
        # according to the following link the scaling and mean computations do not detach the gradient.
        # https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
        integral = gm.integrate(x).view(gm.n_batch(x), gm.n_layers(x), 1)
        if not self.per_mixture_norm:
            integral = torch.mean(integral, dim=0, keepdim=True)
            if not self.per_layer_norm:
                integral = torch.mean(integral, dim=1, keepdim=True)

        weights = gm.weights(x)
        positions = gm.positions(x)
        covariances = gm.covariances(x)

        weights = weights / (integral + 0.0001)
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