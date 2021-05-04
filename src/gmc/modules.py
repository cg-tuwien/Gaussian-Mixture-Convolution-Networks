import math
import typing

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter as TensorboardWriter

import gmc.mixture as gm
import gmc.fitting
import gmc.render
import gmc.inout as gmio
import gmc.mat_tools as mat_tools
import gmc.cpp.gm_vis.gm_vis as gm_vis


class ConvolutionConfig:
    def __init__(self):
        pass

class Convolution(torch.nn.modules.Module):
    def __init__(self, config: ConvolutionConfig, n_layers_in: int, n_layers_out: int, n_kernel_components: int = 4, n_dims: int = 2,
                 position_range: float = 1, covariance_range: float = 0.25, weight_sd=0.1, weight_mean=0.0,
                 learn_positions: bool = True, learn_covariances: bool = True,
                 covariance_epsilon: float = 0.05):
        super(Convolution, self).__init__()
        self.config = config
        self.n_layers_in = n_layers_in
        self.n_layers_out = n_layers_out
        self.n_kernel_components = n_kernel_components
        self.n_dims = n_dims
        self.position_range = position_range
        self.covariance_range = covariance_range
        self.weight_sd = weight_sd
        self.weight_mean = weight_mean
        self.covariance_epsilon = covariance_epsilon
        self.learn_positions = learn_positions
        self.learn_covariances = learn_covariances

        self.weights = torch.nn.modules.ParameterList()
        self.positions = torch.nn.modules.ParameterList()
        self.covariance_factors = torch.nn.modules.ParameterList()

        # todo: probably can optimise performance by putting kernels into their own dimension (currently as a list)
        for i in range(self.n_layers_out):
            # positive mean produces a rather positive gm. i believe this is a better init
            weights = torch.randn(1, n_layers_in, n_kernel_components, 1, dtype=torch.float32)
            self.weights.append(torch.nn.Parameter(weights))

            if self.learn_positions and False:
                positions = torch.rand(1, n_layers_in, n_kernel_components, n_dims, dtype=torch.float32) * 2 - 1
            elif self.n_dims == 2:
                assert (self.n_dims == 2)
                angles = torch.arange(0, 2 * math.pi, 2 * math.pi / (n_kernel_components - 1))
                xes = torch.cat((torch.zeros(1, dtype=torch.float), torch.sin(angles)), dim=0)
                yes = torch.cat((torch.zeros(1, dtype=torch.float), torch.cos(angles)), dim=0)
                positions = torch.cat((xes.view(-1, 1), yes.view(-1, 1)), dim=1)
                positions = positions.view(1, 1, n_kernel_components, 2).repeat((1, n_layers_in, 1, 1))
            else:
                assert (self.n_dims == 3)
                # uniform sphere distribution + one in the middle
                # https://stats.stackexchange.com/questions/7977/how-to-generate-uniformly-distributed-points-on-the-surface-of-the-3-d-unit-sphe
                z = torch.rand(1, n_layers_in, n_kernel_components, 1, dtype=torch.float32) * 2 - 1
                theta = torch.rand(1, n_layers_in, n_kernel_components, 1, dtype=torch.float32) * 2 * 3.14159265358979 - 3.14159265358979
                x = torch.sin(theta) * torch.sqrt(1 - z * z)
                y = torch.cos(theta) * torch.sqrt(1 - z * z)
                positions = torch.cat((x, y, z), dim=3)
                # positions = positions / torch.norm(positions, dim=-1, keepdim=True)
                # + one in the middle
                positions[:, :, 0, :] = 0
            self.positions.append(torch.nn.Parameter(positions))

            # initialise with a rather round covariance matrix
            # a psd matrix can be generated with A A'. we learn A and generate a pd matrix via  A A' + eye * epsilon
            covariance_factors = torch.rand(1, n_layers_in, n_kernel_components, n_dims, n_dims, dtype=torch.float32) * 2 - 1
            cov_rand_factor = 0.05
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

    def kernel(self, index: int) -> Tensor:
        # a psd matrix can be generated with AA'. we learn A and generate a pd matrix via  AA' + eye * epsilon
        weights = self.weights[index] * self.weight_sd + self.weight_mean

        A = self.covariance_factors[index]
        covariances = A @ A.transpose(-1, -2)
        covariances = covariances + torch.eye(self.n_dims, dtype=torch.float32, device=A.device) * self.covariance_epsilon * max(1.0, torch.max(covariances.detach().abs()).item())
        # kernel = torch.cat((self.weights[index], self.positions[index], covariances.view(1, self.n_layers_in, self.n_kernel_components, self.n_dims * self.n_dims)), dim=-1)

        kernel = gm.pack_mixture(weights, self.positions[index] * self.position_range, covariances * self.covariance_range)

        assert gm.is_valid_mixture(kernel)
        return kernel

    def regularisation_loss(self) -> Tensor:
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

    def weight_decay_loss(self) -> Tensor:
        """
        USAGE: use a simple stochastic optimiser! if you use something more elaborate with moments (e.g. Adam), then consider using a separate optimiser as the moments don't play well with weight decay.
        Read more here: https://arxiv.org/abs/1711.05101 and
        https://towardsdatascience.com/weight-decay-l2-regularization-90a9e17713cd (towards the bottom of section "Is L2 Regularization and Weight Decay the same thing?")

        weights are decayed to 0, positions not at all, covariance towards the identity
        """
        n_dims = gm.n_dimensions(self.kernel(0))
        eye = torch.eye(n_dims, device=self.weights[0].device, dtype=self.weights[0].dtype).view(1, 1, 1, n_dims, n_dims)
        loss = torch.zeros(1, dtype=torch.float32, device=self.weights[0].device)
        for k in range(self.n_layers_out):
            A = self.covariance_factors[k]
            covariances = A @ A.transpose(-1, -2)
            temp1 = (self.weights[k] * self.weights[k]).mean()
            temp2 = covariances - eye
            loss = loss + (temp2 * temp2).mean() + temp1

        return loss

    def debug_render(self, position_range: float = None, image_size: int = 80, clamp: typing.Tuple[float, float] = (-0.3, 0.3)):
        if position_range is None:
            position_range = self.position_range * 2

        images = list()
        for i in range(min(self.n_layers_out, 5)):
            kernel = self.kernel(i)
            assert kernel.shape[0] == 1
            l = min(gm.n_layers(kernel), 5)
            kernel_rendering = gmc.render.render(kernel, torch.zeros(1, 1, device=kernel.device), layers=(0, l),
                                         x_low=-position_range*1.25, x_high=position_range*1.25, y_low=-position_range*1.25, y_high=position_range*1.25, width=image_size, height=image_size)
            images.append(kernel_rendering)
        images = torch.cat(images, dim=1)
        images = gmc.render.colour_mapped(images.cpu().numpy(), clamp[0], clamp[1])
        return images[:, :, :3]

    def debug_render3d(self, image_size: int = 80, clamp: typing.Tuple[float, float] = (-0.3, 0.3), camera: typing.Optional[typing.Dict] = None) -> Tensor:
        vis = gm_vis.GMVisualizer(False, image_size, image_size)
        if camera is not None:
            vis.set_camera_lookat(**camera)
        else:
            vis.set_camera_auto(True)
        vis.set_density_rendering(True)
        vis.set_density_range_manual(clamp[0], clamp[1])

        images = list()
        for i in range(self.n_layers_out):
            kernel = self.kernel(i)
            assert kernel.shape[0] == 1
            kernel_rendering = gmc.render.render3d(kernel, width=image_size, height=image_size, gm_vis_object=vis)
            images.append(kernel_rendering)

        vis.finish()
        images = torch.cat(images, dim=1)
        return images[:, :, :3]

    def debug_save3d(self, base_name: str):
        for i in range(self.n_layers_out):
            kernel = self.kernel(i)
            assert kernel.shape[0] == 1

            gmio.write_gm_to_ply2(kernel, f"{base_name}_k{i}")

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
        self.fitting_method = gmc.fitting.fixed_point_and_tree_hem
        self.fitting_config = gmc.fitting.Config()


class ReLUFitting(torch.nn.modules.Module):
    def __init__(self, config: ReLUFittingConfig, n_layers: int, n_output_gaussians: int, n_input_gaussians: int = -1):
        super(ReLUFitting, self).__init__()
        self.config = config
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

        last_in = gmc.render.render(self.last_in[0], self.last_in[1], batches=(0, 1), layers=(0, 5),
                                    x_low=position_range[0], y_low=position_range[1], x_high=position_range[2], y_high=position_range[3],
                                    width=image_size, height=image_size)
        target = gmc.render.render_with_relu(self.last_in[0], self.last_in[1], batches=(0, 1), layers=(0, 5),
                                             x_low=position_range[0], y_low=position_range[1], x_high=position_range[2], y_high=position_range[3],
                                             width=image_size, height=image_size)
        prediction = gmc.render.render(self.last_out[0], self.last_out[1], batches=(0, 1), layers=(0, 5),
                                       x_low=position_range[0], y_low=position_range[1], x_high=position_range[2], y_high=position_range[3],
                                       width=image_size, height=image_size)
        images = [last_in, target, prediction]
        images = torch.cat(images, dim=1)
        images = gmc.render.colour_mapped(images.cpu().numpy(), clamp[0], clamp[1])
        return images[:, :, :3]

    def debug_render3d(self, image_size: int = 80, clamp: typing.Tuple[float, float] = (-0.1, 0.1), camera: typing.Optional[typing.Dict] = None):
        vis = gm_vis.GMVisualizer(False, image_size, image_size)
        if camera is not None:
            vis.set_camera_lookat(**camera)
        else:
            vis.set_camera_auto(True)
        vis.set_density_rendering(True)
        vis.set_density_range_manual(clamp[0], clamp[1])

        last_in = gmc.render.render3d(self.last_in[0], batches=(0, 1), layers=(0, None), gm_vis_object=vis)
        prediction = gmc.render.render3d(self.last_out[0], batches=(0, 1), layers=(0, None), gm_vis_object=vis)
        vis.finish()

        images = torch.cat([last_in, prediction], dim=1)
        return images[:, :, :3]

    def debug_save3d(self, base_name: str, n_batch_samples: int = 1, n_layers: int = None):
        gmio.write_gm_to_ply2(self.last_in[0][0:n_batch_samples, 0:n_layers], f"{base_name}_in")
        gmio.write_gm_to_ply2(self.last_out[0][0:n_batch_samples, 0:n_layers], f"{base_name}_out")


class Dropout(torch.nn.modules.Module):
    """
    drops a certain percentage of layers (i.e., sets the weights to zero)
    """
    def __init__(self, drop_percentage: float):
        super(Dropout, self).__init__()
        self.drop_percentage = drop_percentage

    def forward(self, x: typing.Tuple[Tensor, typing.Optional[Tensor]]) -> typing.Tuple[Tensor, Tensor]:
        x_constant = x[1]
        x_gm = x[0]

        assert gm.is_valid_mixture(x_gm)

        if self.training:
            n_channels = gm.n_layers(x_gm)
            n_batch = gm.n_batch(x_gm)
            n_dims = gm.n_dimensions(x_gm)
            dropouts = (torch.rand(n_batch, n_channels, dtype=torch.float32, device=x_gm.device) < self.drop_percentage).unsqueeze(-1)

            weights = torch.where(dropouts,
                                  torch.scalar_tensor(0.0, dtype=torch.float32, device=x_gm.device),
                                  gm.weights(x_gm) * (1.0 / (1 - self.drop_percentage)))

            positions = torch.where(dropouts.unsqueeze(-1),
                                    torch.scalar_tensor(0, dtype=torch.float32, device=x_gm.device),
                                    gm.positions(x_gm))

            covs = torch.where(dropouts.unsqueeze(-1).unsqueeze(-1),
                               torch.eye(n_dims, dtype=torch.float32, device=x_gm.device).view(1, 1, 1, n_dims, n_dims),
                               gm.covariances(x_gm))

            y_gm = gm.pack_mixture(weights, positions, covs)

            return y_gm, x_constant

        return x_gm, x_constant


class CovScaleNorm(torch.nn.modules.Module):
    """
    this norm scales the covariances and positions so that the average covariance trace is n_dimensions.
    scaling is uniform in all spatial directions (x, y, and z)
    """
    def __init__(self, n_layers: int, batch_norm: bool = True):
        """
        Parameters:
            batch_norm (bool): True if this is a normal batch norm, False if normalisation should be performed per training sample (e.g., for the input).
        """
        super(CovScaleNorm, self).__init__()
        self.batch_norm = batch_norm
        self.register_buffer("averaged_cov_trace", torch.ones(n_layers))

    def forward(self, x: typing.Tuple[Tensor, typing.Optional[Tensor]]) -> typing.Tuple[Tensor, Tensor]:
        # according to the following link the scaling and mean computations do not detach the gradient.
        # https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html

        x_constant = x[1]
        x_gm = x[0]

        assert gm.is_valid_mixture(x_gm)

        traces = mat_tools.trace(gm.covariances(x_gm))
        n_batch = gm.n_batch(x_gm)
        if self.batch_norm:
            avg_cov_trace = torch.mean(traces, dim=(0, 2))
            if self.training:
                alpha = min(n_batch / 1000, 0.2)
                self.averaged_cov_trace = ((1.0 - alpha) * self.averaged_cov_trace + alpha * avg_cov_trace).detach()
            n_batch = 1
            avg_cov_trace = self.averaged_cov_trace.detach()
        else:
            avg_cov_trace = torch.mean(traces, dim=(2, ))

        scaling_factor = avg_cov_trace / gm.n_dimensions(x_gm)

        assert (scaling_factor > 0).all().item();

        scaling_factor = (1 / scaling_factor).view(n_batch, gm.n_layers(x_gm), 1)
        scaling_factor = torch.sqrt(scaling_factor)

        assert not torch.any(torch.isnan(scaling_factor))
        assert not torch.any(torch.isinf(scaling_factor))

        y_gm = gm.spatial_scale(x_gm, scaling_factor)

        if x_constant is None:
            y_constant = torch.zeros(1, 1, device=x_gm.device)
        else:
            y_constant = x_constant

        return y_gm, y_constant


class BatchNorm(torch.nn.modules.Module):
    """
    this norm scales the weights so that that variance of the gm is 1 within pos_min and pos_max.
    """
    def __init__(self, n_layers: int, batch_norm: bool = True, learn_scaling: bool = True):
        """
        Parameters:
        batch_norm (bool): True if this is a normal batch norm, False if normalisation should be performed per training sample (e.g., for the input).
        n_layers (optional int): Number of layers if a learnable scaling parameter should be used, None otherwise.
        """
        super(BatchNorm, self).__init__()
        self.batch_norm = batch_norm
        if learn_scaling:
            self.learnable_scaling = torch.nn.Parameter(torch.ones(n_layers))
        else:
            self.learnable_scaling = None

        self.register_buffer("averaged_channel_sd", torch.ones(n_layers))

    def forward(self, x: typing.Tuple[Tensor, typing.Optional[Tensor]]) -> typing.Tuple[Tensor, Tensor]:
        # according to the following link the scaling and mean computations do not detach the gradient.
        # https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html

        x_gm = x[0]
        x_constant = x[1]

        if x_constant is None:
            x_constant = torch.zeros(1, 1, device=x_gm.device)

        assert gm.is_valid_mixture_and_constant(x_gm, x_constant)

        n_batch = gm.n_batch(x_gm)
        n_channels = gm.n_layers(x_gm)
        n_dim = gm.n_dimensions(x_gm)
        n_sampling_positions = 10000 // n_batch

        # sampling position per training sample
        pos_min = torch.min(gm.positions(x_gm).view(n_batch, -1, n_dim), dim=1)[0].view(n_batch, 1, 1, n_dim).detach()
        pos_max = torch.max(gm.positions(x_gm).view(n_batch, -1, n_dim), dim=1)[0].view(n_batch, 1, 1, n_dim).detach()

        sampling_positions = torch.rand(n_batch, n_channels, n_sampling_positions, n_dim, device=x_gm.device) * (pos_max - pos_min) + pos_min
        sample_values = gm.evaluate(x_gm, sampling_positions) + x_constant
        # channel_sd, channel_mean = torch.std_mean(sample_values.transpose(0, 1).reshape(n_channels, n_batch * n_sampling_positions), dim=1)
        if self.batch_norm:
            channel_sd, channel_mean = torch.std_mean(sample_values, dim=(0, 2))
            if self.training:
                alpha = min(n_batch / 1000, 0.1)
                self.averaged_channel_sd = ((1.0 - alpha) * self.averaged_channel_sd + alpha * channel_sd).detach()
            channel_sd = self.averaged_channel_sd.detach()
        else:
            channel_sd, channel_mean = torch.std_mean(sample_values, dim=2)

        channel_sd = torch.max(channel_sd, torch.tensor([0.001], dtype=torch.float, device=x_gm.device))
        scaling_factor = 1 / channel_sd
        if self.learnable_scaling is not None:
            scaling_factor = scaling_factor * self.learnable_scaling

        scaling_factor = scaling_factor.view(-1, n_channels, 1)
        assert not torch.any(torch.isnan(scaling_factor))
        assert not torch.any(torch.isinf(scaling_factor))

        new_weights = gm.weights(x_gm) * scaling_factor

        return gm.pack_mixture(new_weights, gm.positions(x_gm), gm.covariances(x_gm)), x_constant - channel_mean

