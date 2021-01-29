import typing
import time

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter as TensorboardWriter

import gmc.mixture as gm
from gmc.cpp.extensions.bvh_mhem_fit import binding as cppBvhMhemFit


class Config:
    def __init__(self, n_reduction: int=4):
        self.n_reduction = n_reduction


def fixed_point_and_bvh_mhem(mixture: Tensor, constant: Tensor, n_components: int, config: Config = Config(), tensorboard: TensorboardWriter = None) -> typing.Tuple[Tensor, Tensor, typing.List[Tensor]]:
    if n_components < 0:
        initial_fitting = initial_approx_to_relu(mixture, constant)
        fitting, ret_const = fixed_point_iteration_to_relu(mixture, constant, initial_fitting)
        return fitting, ret_const, [initial_fitting]

    if tensorboard is not None:
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    initial_fitting = initial_approx_to_relu(mixture, constant)

    if tensorboard is not None:
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        tensorboard.add_scalar(f"50.1 fitting {mixture.shape} -> {n_components} initial_approx_to_relu time =", t1 - t0, 0)

    fp_fitting, ret_const = fixed_point_iteration_to_relu(mixture, constant, initial_fitting)

    if tensorboard is not None:
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        tensorboard.add_scalar(f"50.2 fitting {mixture.shape} -> {n_components} fixed_point_iteration_to_relu time =", t2 - t1, 0)

    fitting = cppBvhMhemFit.apply(fp_fitting, n_components, config.n_reduction)

    if tensorboard is not None:
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        tensorboard.add_scalar(f"50.5 fitting {mixture.shape} -> {n_components} bvh_mhem_fit time=", t3 - t2, 0)

    return fitting, ret_const, [initial_fitting, fp_fitting]


def initial_approx_to_relu(mixture: Tensor, constant: Tensor) -> Tensor:
    device = mixture.device
    relu_const = constant.where(constant > 0, torch.zeros(1, device=device))
    weights = gm.weights(mixture)
    new_weights = weights.where(weights + constant.unsqueeze(-1) > 0, -relu_const.unsqueeze(-1))
    return gm.pack_mixture(new_weights, gm.positions(mixture), gm.covariances(mixture))


def fixed_point_iteration_to_relu(target_mixture: Tensor, target_constant: Tensor, fitting_mixture: Tensor, n_iter: int = 1) -> typing.Tuple[Tensor, Tensor]:
    assert gm.is_valid_mixture_and_constant(target_mixture, target_constant)
    assert gm.is_valid_mixture(fitting_mixture)
    assert gm.n_batch(target_mixture) == gm.n_batch(fitting_mixture)
    assert gm.n_layers(target_mixture) == gm.n_layers(fitting_mixture)
    assert gm.n_dimensions(target_mixture) == gm.n_dimensions(fitting_mixture)
    assert target_mixture.device == fitting_mixture.device

    weights = gm.weights(fitting_mixture)
    positions = gm.positions(fitting_mixture)
    covariances = gm.covariances(fitting_mixture)
    device = fitting_mixture.device

    # todo: make transfer fitting function configurable. e.g. these two can be replaced by leaky relu or softplus (?, might break if we have many overlaying Gs)
    ret_const = target_constant.where(target_constant > 0, torch.zeros(1, device=device))
    b = gm.evaluate(target_mixture, positions) + target_constant.unsqueeze(-1)
    b = b.where(b > 0, torch.zeros(1, device=device)) - ret_const.unsqueeze(-1)

    x = weights.abs() + 0.05

    for i in range(n_iter):
        x = x.abs()
        new_mixture = gm.pack_mixture(x, positions, covariances)
        x = x * b / (gm.evaluate(new_mixture, positions) + 0.05)

    return gm.pack_mixture(x, positions, covariances), ret_const


def generate_random_sampling(m: Tensor, n: int) -> Tensor:
    covariance_adjustment = torch.sqrt(torch.diagonal(gm.covariances(m.detach()), dim1=-2, dim2=-1))
    position_max, _ = torch.max(gm.positions(m.detach()) + covariance_adjustment, dim=2, keepdim=True)
    position_min, _ = torch.min(gm.positions(m.detach()) - covariance_adjustment, dim=2, keepdim=True)
    sampling = torch.rand(gm.n_batch(m), gm.n_layers(m), n, gm.n_dimensions(m), device=m.device)
    sampling *= position_max - position_min
    sampling += position_min
    return sampling


def mse(target_mixture: Tensor, target_constant: Tensor, fitting_mixture: Tensor, fitting_constant: Tensor, n_test_points: int = 500) -> float:
    xes = generate_random_sampling(target_mixture, n_test_points)
    ground_truth = gm.evaluate_with_activation_fun(target_mixture, target_constant, xes)
    gt_mean = ground_truth.mean()
    gt_sd = ground_truth.std()
    ground_truth = (ground_truth - gt_mean) / gt_sd

    fitting = gm.evaluate(fitting_mixture, xes) + fitting_constant.unsqueeze(-1)
    fitting = (fitting - gt_mean) / gt_sd
    return ((fitting - ground_truth)**2).mean().item()
