import typing
import time

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter as TensorboardWriter

import gmc.mixture as gm
import gmc.mat_tools as mat_tools


class Config:
    REPRESENTATIVE_SELECT_MODE_TOP_INTEGRALS: int = 0
    REPRESENTATIVE_SELECT_MODE_RANDOM_TOP: int = 1

    def __init__(self, n_iterations: int = 1, KL_divergence_threshold: float = 1.5, representative_select_mode: int = REPRESENTATIVE_SELECT_MODE_TOP_INTEGRALS):
        self.n_iterations = n_iterations
        self.KL_divergence_threshold = KL_divergence_threshold
        self.representative_select_mode = representative_select_mode


def fixed_point_and_mhem(mixture: Tensor, constant: Tensor, n_components: int, config: Config = Config(), tensorboard: TensorboardWriter = None) -> typing.Tuple[Tensor, Tensor, typing.List[Tensor]]:
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
    reduced_fitting = representative_select_for_relu(fp_fitting.detach(), n_components, config)
    if tensorboard is not None:
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        tensorboard.add_scalar(f"50.3 fitting {mixture.shape} -> {n_components} representative_select_for_relu time =", t3 - t2, 0)
    fitting = mhem_fit_a_to_b(reduced_fitting, fp_fitting, config, tensorboard)
    if tensorboard is not None:
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        tensorboard.add_scalar(f"50.4 fitting {mixture.shape} -> {n_components} mhem_fit_a_to_b time=", t4 - t3, 0)

    return fitting, ret_const, [initial_fitting, fp_fitting, reduced_fitting]


def fixed_point_only(mixture: Tensor, constant: Tensor, n_components: int, config: Config = Config()) -> typing.Tuple[Tensor, Tensor, typing.List[Tensor]]:
    if n_components < 0:
        initial_fitting = initial_approx_to_relu(mixture, constant)
        fitting, ret_const = fixed_point_iteration_to_relu(mixture, constant, initial_fitting)
        return fitting, ret_const, [initial_fitting]

    initial_fitting = initial_approx_to_relu(mixture, constant)
    reduced_fitting = representative_select_for_relu(initial_fitting, n_components, config)
    fp_fitting, ret_const = fixed_point_iteration_to_relu(mixture, constant, reduced_fitting)
    # fitting = mhem_fit_a_to_b(reduced_fitting, fitting)
    return fp_fitting, ret_const, [initial_fitting, reduced_fitting]


def mhem_and_fixed_point(mixture: Tensor, constant: Tensor, n_components: int, config: Config = Config()) -> typing.Tuple[Tensor, Tensor, typing.List[Tensor]]:
    if n_components < 0:
        initial_fitting = initial_approx_to_relu(mixture, constant)
        fitting, ret_const = fixed_point_iteration_to_relu(mixture, constant, initial_fitting)
        return fitting, ret_const, [initial_fitting]

    initial_fitting = initial_approx_to_relu(mixture.detach(), constant.detach())
    reduced_fitting = representative_select_for_relu(initial_fitting, n_components, config)
    mhem_fitting = mhem_fit_a_to_b(reduced_fitting, initial_fitting, config)
    fp_fitting, ret_const = fixed_point_iteration_to_relu(mixture, constant, mhem_fitting)
    return fp_fitting, ret_const, [initial_fitting, reduced_fitting, mhem_fitting]


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


def mixture_to_double_gmm(mixture: Tensor) -> typing.Tuple[Tensor, Tensor, Tensor]:
    component_integrals = gm.integrate_components(mixture)
    abs_integrals = component_integrals.abs().sum(dim=2, keepdim=True)
    abs_integrals = abs_integrals.where(abs_integrals > 0.05, 0.05 * torch.ones_like(abs_integrals))

    new_weights = gm.weights(mixture) / abs_integrals
    target_double_gmm = gm.pack_mixture(new_weights, gm.positions(mixture), gm.covariances(mixture))
    return target_double_gmm, component_integrals, abs_integrals


def calc_likelihoods(target: Tensor, fitting: Tensor) -> Tensor:
    # index i -> target
    # index s -> fitting
    n_batch = gm.n_batch(target)
    n_layers = gm.n_layers(target)
    n_target_components = gm.n_components(target)
    n_fitting_components = gm.n_components(fitting)
    n_dims = gm.n_dimensions(target)
    n_virtual_points = n_fitting_components

    target_weights = gm.weights(target).abs()
    target_positions = gm.positions(target)
    target_covariances = gm.covariances(target)
    target_normal_amplitudes = gm.normal_amplitudes(target_covariances)

    # fitting weights are not used here, only in mhem_algorithm()
    fitting_positions = gm.positions(fitting)
    fitting_covariances = gm.covariances(fitting)
    fitting_normal_amplitudes = gm.normal_amplitudes(fitting_covariances)

    # todo: n_virtual_points should be separate for +-
    target_n_virtual_points = n_virtual_points * target_weights / target_normal_amplitudes

    # preiner equation 9
    gaussian_values = gm.evaluate_componentwise(gm.pack_mixture(fitting_normal_amplitudes, fitting_positions, fitting_covariances), target_positions)
    exp_values = torch.exp(-0.5 * mat_tools.batched_trace(mat_tools.inverse(fitting_covariances).view(n_batch, n_layers, 1, n_fitting_components, n_dims, n_dims) @
                                                          target_covariances.view(n_batch, n_layers, n_target_components, 1, n_dims, n_dims)))

    almost_likelihoods = gaussian_values * exp_values

    return torch.pow(almost_likelihoods, target_n_virtual_points.unsqueeze(-1))


def calc_KL_divergence(target: Tensor, fitting: Tensor) -> Tensor:
    # index i -> target
    # index s -> fitting

    target_positions = gm.positions(target).unsqueeze(3)
    target_covariances = gm.covariances(target).unsqueeze(3)

    fitting_positions = gm.positions(fitting).unsqueeze(2)
    fitting_covariances = gm.covariances(fitting).unsqueeze(2)
    fitting_covariances_inversed = mat_tools.inverse(fitting_covariances)

    p_diff = target_positions - fitting_positions
    mahalanobis_distance = torch.sqrt(p_diff.unsqueeze(-2) @ fitting_covariances_inversed @ p_diff.unsqueeze(-1)).squeeze(dim=-1).squeeze(dim=-1)
    trace = mat_tools.batched_trace(fitting_covariances_inversed @ target_covariances)
    logarithm = torch.log(target_covariances.det() / fitting_covariances.det())
    KL_divergence = 0.5 * (mahalanobis_distance + trace - gm.n_dimensions(target) - logarithm)

    return KL_divergence


def representative_select_for_relu(mixture: Tensor, n_components: int, config: Config = Config()) -> Tensor:
    assert n_components > 0
    component_integrals = gm.integrate_components(mixture.detach()).abs()

    n_original_components = gm.n_components(mixture)
    device = mixture.device

    if config.representative_select_mode == config.REPRESENTATIVE_SELECT_MODE_TOP_INTEGRALS:
        # original
        _, sorted_indices = torch.sort(component_integrals, descending=True)
        selection_mixture = mat_tools.my_index_select(mixture, sorted_indices[:, :, :n_components])
    elif config.representative_select_mode == config.REPRESENTATIVE_SELECT_MODE_RANDOM_TOP:
        # random sampling
        _, sorted_indices = torch.sort(component_integrals, descending=True)
        selection_mixture = mat_tools.my_index_select(mixture, sorted_indices[:, :, :min(max(n_components*2, n_original_components // 4), n_original_components)])
        selection_mixture = mat_tools.my_index_select(selection_mixture, torch.randperm(selection_mixture.shape[-2], device=device)[:n_components].view(1, 1, n_components))
    else:
        assert False

    return selection_mixture


def mhem_fit_a_to_b(fitting_mixture: Tensor, target_mixture: Tensor, config: Config = Config(), tensorboard: TensorboardWriter = None) -> Tensor:
    assert gm.is_valid_mixture(fitting_mixture)
    assert gm.is_valid_mixture(target_mixture)
    assert gm.n_batch(target_mixture) == gm.n_batch(fitting_mixture)
    assert gm.n_layers(target_mixture) == gm.n_layers(fitting_mixture)
    assert gm.n_dimensions(target_mixture) == gm.n_dimensions(fitting_mixture)
    assert target_mixture.device == fitting_mixture.device

    n_batch = gm.n_batch(target_mixture)
    n_layers = gm.n_layers(target_mixture)
    n_components_target = gm.n_components(target_mixture)
    n_components_fitting = gm.n_components(fitting_mixture)
    n_dims = gm.n_dimensions(target_mixture)
    device = target_mixture.device

    target_double_gmm, component_integrals, abs_integrals = mixture_to_double_gmm(target_mixture)
    # target_double_gmm, integrals = mixture_to_gmm(target)

    fitting_double_gmm, _, _ = mixture_to_double_gmm(fitting_mixture)

    # log(target_double_gmm, 0, tensor_board_writer, layer=layer)
    # log(fitting_double_gmm, 1, tensor_board_writer, layer=layer)

    assert gm.is_valid_mixture(target_double_gmm)
    for i in range(config.n_iterations):
        assert gm.is_valid_mixture(fitting_double_gmm)

        target_weights = gm.weights(target_double_gmm).unsqueeze(3)
        fitting_weights = gm.weights(fitting_double_gmm).unsqueeze(2)
        sign_match = target_weights.sign() == fitting_weights.sign()
        if tensorboard is not None:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        likelihoods = calc_likelihoods(target_double_gmm.detach(), fitting_double_gmm.detach())
        if tensorboard is not None:
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            tensorboard.add_scalar(f"50.4.1 mhem_fit_a_to_b {target_mixture.shape} -> {gm.n_components(fitting_mixture)} calc_likelihoods time =", t1 - t0, 0)

        KL_divergence = calc_KL_divergence(target_double_gmm.detach(), fitting_double_gmm.detach())
        if tensorboard is not None:
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            tensorboard.add_scalar(f"50.4.2 mhem_fit_a_to_b {target_mixture.shape} -> {gm.n_components(fitting_mixture)} KL_divergence time =", t2 - t1, 0)

        likelihoods = likelihoods * (gm.weights(fitting_double_gmm).abs() / gm.normal_amplitudes(gm.covariances(fitting_double_gmm))).view(n_batch, n_layers, 1, n_components_fitting)
        likelihoods = likelihoods * (KL_divergence < config.KL_divergence_threshold) * sign_match

        likelihoods_sum = likelihoods.sum(3, keepdim=True)
        responsibilities = likelihoods / (likelihoods_sum + 0.00001)

        if tensorboard is not None:
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            tensorboard.add_scalar(f"50.4.3 mhem_fit_a_to_b {target_mixture.shape} -> {gm.n_components(fitting_mixture)} other 1 time =", t3 - t2, 0)
        assert not torch.any(torch.isnan(responsibilities))

        # index i -> target
        # index s -> fitting
        responsibilities = responsibilities * (gm.weights(target_double_gmm).abs() / gm.normal_amplitudes(gm.covariances(target_double_gmm))).unsqueeze(-1)
        newWeights = torch.sum(responsibilities, 2)
        assert not torch.any(torch.isnan(responsibilities))

        assert not torch.any(torch.isnan(newWeights))
        responsibilities = responsibilities / (newWeights + 0.00001).view(n_batch, n_layers, 1, n_components_fitting)
        assert torch.all(responsibilities >= 0)
        assert not torch.any(torch.isnan(responsibilities))
        newPositions = torch.sum(responsibilities.unsqueeze(-1) * gm.positions(target_double_gmm).view(n_batch, n_layers, n_components_target, 1, n_dims), 2)
        assert not torch.any(torch.isnan(newPositions))
        posDiffs = gm.positions(target_double_gmm).view(n_batch, n_layers, n_components_target, 1, n_dims, 1) - newPositions.view(n_batch, n_layers, 1, n_components_fitting, n_dims, 1)
        assert not torch.any(torch.isnan(posDiffs))

        newCovariances = (torch.sum(responsibilities.unsqueeze(-1).unsqueeze(-1) * (gm.covariances(target_double_gmm).view(n_batch, n_layers, n_components_target, 1, n_dims, n_dims) +
                                                                                    posDiffs.matmul(posDiffs.transpose(-1, -2))), 2))
        newCovariances = newCovariances + (newWeights < 0.0001).unsqueeze(-1).unsqueeze(-1) * torch.eye(n_dims, device=device).view(1, 1, 1, n_dims, n_dims) * 0.0001

        assert not torch.any(torch.isnan(newCovariances))

        normal_amplitudes = gm.normal_amplitudes(newCovariances)
        fitting_double_gmm = gm.pack_mixture(newWeights.contiguous() * normal_amplitudes * gm.weights(fitting_double_gmm).sign(), newPositions.contiguous(), newCovariances.contiguous())

        if tensorboard is not None:
            torch.cuda.synchronize()
            t4 = time.perf_counter()
            tensorboard.add_scalar(f"50.4.4 mhem_fit_a_to_b {target_mixture.shape} -> {gm.n_components(fitting_mixture)} other 2 time =", t4 - t3, 0)

    # the following line would have the following effect: scale the fitting result to match the integral of the input exactly. that is bad in case many weak Gs are killed and the remaining weak G is blown up.
    # fitting_double_gmm, _, _, _ = mixture_to_double_gmm(fitting_double_gmm)

    newWeights = gm.weights(fitting_double_gmm) * abs_integrals
    fitting = gm.pack_mixture(newWeights, gm.positions(fitting_double_gmm), gm.covariances(fitting_double_gmm))
    return fitting


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
