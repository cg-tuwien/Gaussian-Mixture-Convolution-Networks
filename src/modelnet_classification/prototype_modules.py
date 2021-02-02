import typing

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter as TensorboardWriter

import gmc.mixture as gm
import gmc.mat_tools as mat_tools
import modelnet_classification.config as Config


class BatchNorm(torch.nn.modules.Module):
    def __init__(self, config: Config, per_mixture_norm: bool = False):
        super(BatchNorm, self).__init__()
        self.per_mixture_norm = per_mixture_norm
        self.config = config

    def forward(self, x: Tensor, x_constant: Tensor = None) -> typing.Tuple[Tensor, Tensor]:
        # this is an adapted batch norm. It scales and centres the gm, but it has nothing to do with variance or mean
        # both of them require a domain, or footprint for the computation, but our gaussians extend to infinity.
        # in a way, variance approaches always zero and the mean approaches x_constant. we can't use them.

        # according to the following link the scaling and mean computations do not detach the gradient.
        # https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
        if x_constant is not None:
            assert gm.is_valid_mixture_and_constant(x, x_constant)
        else:
            assert gm.is_valid_mixture(x)

        abs_x = gm.pack_mixture(torch.max(gm.weights(x), torch.zeros(1, 1, 1, device=x.device)), gm.positions(x), gm.covariances(x))
        integral_abs = gm.integrate(abs_x)
        if not self.per_mixture_norm:
            integral_abs = torch.mean(integral_abs, dim=0, keepdim=True)
            if self.config.bn_mean_over_layers:
                integral_abs = torch.mean(integral_abs, dim=1, keepdim=True)

        weights = gm.weights(x)
        positions = gm.positions(x)
        covariances = gm.covariances(x)
        integral_abs_eps = integral_abs + 0.01
        weights = weights / integral_abs_eps.unsqueeze(-1)
        mixture = gm.pack_mixture(weights, positions, covariances)

        if self.config.bn_constant_computation == Config.BN_CONSTANT_COMPUTATION_ZERO:
            y_constant = torch.zeros(1, 1, device=x.device)
        elif self.config.bn_constant_computation == Config.BN_CONSTANT_COMPUTATION_INTEGRAL:
            y_constant = -gm.integrate(abs_x)
        elif self.config.bn_constant_computation == Config.BN_CONSTANT_COMPUTATION_MEAN_IN_CONST:
            if x_constant is None:
                y_constant = torch.zeros(1, 1, device=x.device)
            elif x_constant.shape[0] == 1:
                # std produces NaN gradients in that case
                # c - c.mean produces always 0 gradients, hence the following is the same.
                y_constant = torch.zeros(1, 1, device=x.device)
            else:
                y_constant = x_constant - x_constant.mean(dim=0, keepdim=True)
                y_constant = y_constant / (y_constant.std(dim=0, keepdim=True, unbiased=False) + 0.001)
        else:
            assert self.config.bn_constant_computation == Config.BN_CONSTANT_COMPUTATION_WEIGHTED
            if x_constant is None:
                y_constant = torch.zeros(1, 1, device=x.device)
            elif x_constant.shape[0] == 1:
                # std produces NaN gradients in that case
                # c - c.mean produces always 0 gradients, hence the following is the same.
                y_constant = torch.zeros(1, 1, device=x.device)
            else:
                y_constant = x_constant - x_constant.mean(dim=0, keepdim=True)
                y_constant = y_constant / (y_constant.std(dim=0, keepdim=True, unbiased=False) + 0.001)

            y_constant = 0.5 * (y_constant - gm.integrate(abs_x))

        return mixture, y_constant


class MaxPooling(torch.nn.modules.Module):
    def __init__(self, n_output_gaussians: int = 10):
        super(MaxPooling, self).__init__()
        self.n_output_gaussians = n_output_gaussians

    def forward(self, x: Tensor) -> Tensor:
        sorted_indices = torch.argsort(gm.integrate_components(x.detach()), dim=2, descending=True)
        sorted_mixture = mat_tools.my_index_select(x, sorted_indices)

        n_output_gaussians = min(self.n_output_gaussians, gm.n_components(x))
        return sorted_mixture[:, :, :n_output_gaussians, :]

