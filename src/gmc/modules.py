import math
import typing

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter as TensorboardWriter

import gmc.mixture as gm
import gmc.image_tools as madam_imagetools
import gmc.fitting

import gmc.cpp.gm_vis.gm_vis as gm_vis


class ConvolutionConfig:
    def __init__(self):
        pass


class Convolution(torch.nn.modules.Module):
    def __init__(self, config: ConvolutionConfig, n_layers_in: int, n_layers_out: int, n_kernel_components: int = 4, n_dims: int = 2,
                 position_range: float = 1, covariance_range: float = 0.25, weight_sd=0.1, weight_mean=0.0,
                 learn_positions: bool = True, learn_covariances: bool = True,
                 covariance_epsilon: float = 0.0001):
        super(Convolution, self).__init__()
        self.config = config
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
                positions[..., 2] = 0
            elif self.n_dims == 2:
                assert (self.n_dims == 2)
                angles = torch.arange(0, 2 * math.pi, 2 * math.pi / (n_kernel_components - 1))
                xes = torch.cat((torch.zeros(1, dtype=torch.float), torch.sin(angles)), dim=0)
                yes = torch.cat((torch.zeros(1, dtype=torch.float), torch.cos(angles)), dim=0)
                positions = torch.cat((xes.view(-1, 1), yes.view(-1, 1)), dim=1)
                positions = positions.view(1, 1, n_kernel_components, 3).repeat((1, n_layers_in, 1, 1))
            else:
                assert (self.n_dims == 3)
                angles = torch.arange(0, 2 * math.pi, 2 * math.pi / (n_kernel_components - 1))
                xes = torch.cat((torch.zeros(1, dtype=torch.float), torch.sin(angles)), dim=0)
                yes = torch.cat((torch.zeros(1, dtype=torch.float), torch.cos(angles)), dim=0)
                zes = torch.zeros_like(xes) # todo fix for real 3d
                positions = torch.cat((xes.view(-1, 1), yes.view(-1, 1), zes.view(-1, 1)), dim=1)
                positions = positions.view(1, 1, n_kernel_components, 3).repeat((1, n_layers_in, 1, 1))
            self.positions.append(torch.nn.Parameter(positions))

            # initialise with a rather round covariance matrix
            # a psd matrix can be generated with A A'. we learn A and generate a pd matrix via  A A' + eye * epsilon
            covariance_factors = torch.rand(1, n_layers_in, n_kernel_components, n_dims, n_dims, dtype=torch.float32) * 2 - 1
            cov_rand_factor = 0.3
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
            kernel_rendering = gm.render(kernel, torch.zeros(1, 1, device=kernel.device),
                                         x_low=-position_range*1.25, x_high=position_range*1.25, y_low=-position_range*1.25, y_high=position_range*1.25, width=image_size, height=image_size)
            images.append(kernel_rendering)
        images = torch.cat(images, dim=1)
        images = madam_imagetools.colour_mapped(images.cpu().numpy(), clamp[0], clamp[1])
        return images[:, :, :3]

    def debug_render3d(self, image_size: int = 80, clamp: typing.Tuple[float, float] = (-0.3, 0.3)):
        vis = gm_vis.GMVisualizer(False, image_size, image_size)
        vis.set_camera_auto(True)
        vis.set_density_rendering(True)
        vis.set_density_range_manual(clamp[0], clamp[1])

        images = list()
        for i in range(self.n_layers_out):
            kernel = self.kernel(i)
            assert kernel.shape[0] == 1
            kernel_rendering = gm.render3d(kernel, width=image_size, height=image_size, gm_vis_object=vis)
            images.append(kernel_rendering)

        vis.finish()
        images = torch.cat(images, dim=1)
        return images[:, :, :3]

    def forward(self, x: Tensor, x_constant: Tensor) -> typing.Tuple[Tensor, Tensor]:
        assert gm.is_valid_mixture_and_constant(x, x_constant)
        out_mixtures = []
        out_constants = []
        out_mixture_shape = list(x.shape)
        out_mixture_shape[1] = 1
        out_mixture_shape[2] = -1

        for i in range(self.n_layers_out):
            kernel = self.kernel(i)
            m = gm.convolve(x, kernel)
            out_mixtures.append(m.view(out_mixture_shape))
            out_constants.append((x_constant * gm.integrate(kernel)).sum(dim=1, keepdim=True))

        return torch.cat(out_mixtures, dim=1), torch.cat(out_constants, dim=1)


class ReLUFittingConfig:
    def __init__(self):
        self.fitting_method = gmc.fitting.fixed_point_and_bvh_mhem
        self.fitting_config = gmc.fitting.Config()


class ReLUFitting(torch.nn.modules.Module):
    def __init__(self, config: ReLUFittingConfig, layer_id: str, n_layers: int, n_output_gaussians: int, n_input_gaussians: int = -1):
        super(ReLUFitting, self).__init__()
        self.config = config
        self.layer_id = layer_id
        self.n_layers = n_layers
        self.n_input_gaussians = n_input_gaussians
        self.n_output_gaussians = n_output_gaussians

        self.last_in = None
        self.last_out = None

    def forward(self, x_m: Tensor, x_constant: Tensor, tensorboard: TensorboardWriter = None) -> typing.Tuple[Tensor, Tensor]:
        y_m, y_constant, _ = self.config.fitting_method(x_m, x_constant, self.n_output_gaussians, self.config.fitting_config, tensorboard)

        self.last_in = (x_m.detach(), x_constant.detach())
        self.last_out = (y_m.detach(), y_constant.detach())
        return y_m, y_constant

    def debug_render(self, position_range: typing.Tuple[float, float, float, float] = None, image_size: int = 80, clamp: typing.Tuple[float, float] = None):
        if position_range is None:
            covariance_adjustment = torch.sqrt(torch.diagonal(gm.covariances(self.last_in[0]), dim1=-2, dim2=-1))
            position_max = gm.positions(self.last_in[0]) + covariance_adjustment
            position_max_x = torch.max(position_max[:, :, :, 0]).item()
            position_max_y = torch.max(position_max[:, :, :, 1]).item()
            position_min = gm.positions(self.last_in[0]) - covariance_adjustment
            position_min_x = torch.min(position_min[:, :, :, 0]).item()
            position_min_y = torch.min(position_min[:, :, :, 1]).item()
            position_range = [position_min_x, position_min_y, position_max_x, position_max_y]

        if clamp is None:
            max_weight = gm.weights(self.last_out[0]).max().item()
            min_weight = gm.weights(self.last_out[0]).min().item()
            abs_diff = max_weight - min_weight
            clamp = (min_weight - abs_diff * 2, min_weight + abs_diff * 2)

        last_in = gm.render(self.last_in[0], self.last_in[1], batches=(0, 1), layers=(0, None),
                            x_low=position_range[0], y_low=position_range[1], x_high=position_range[2], y_high=position_range[3],
                            width=image_size, height=image_size)
        target = gm.render_with_relu(self.last_in[0], self.last_in[1], batches=(0, 1), layers=(0, None),
                                     x_low=position_range[0], y_low=position_range[1], x_high=position_range[2], y_high=position_range[3],
                                     width=image_size, height=image_size)
        prediction = gm.render(self.last_out[0], self.last_out[1], batches=(0, 1), layers=(0, None),
                               x_low=position_range[0], y_low=position_range[1], x_high=position_range[2], y_high=position_range[3],
                               width=image_size, height=image_size)
        images = [last_in, target, prediction]
        images = torch.cat(images, dim=1)
        images = madam_imagetools.colour_mapped(images.cpu().numpy(), clamp[0], clamp[1])
        return images[:, :, :3]

    def debug_render3d(self, image_size: int = 80, clamp: typing.Tuple[float, float] = (-2, 2)):
        vis = gm_vis.GMVisualizer(False, image_size, image_size)
        vis.set_camera_auto(True)
        vis.set_density_rendering(True)
        vis.set_density_range_manual(clamp[0], clamp[1])
        last_in = gm.render3d(self.last_in[0], batches=(0, 1), layers=(0, None), gm_vis_object=vis)
        prediction = gm.render3d(self.last_out[0], batches=(0, 1), layers=(0, None), gm_vis_object=vis)
        vis.finish()

        images = torch.cat([last_in, prediction], dim=1)
        return images[:, :, :3]
