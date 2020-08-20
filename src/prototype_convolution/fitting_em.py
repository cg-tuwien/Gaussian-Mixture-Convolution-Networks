import typing

import torch
from torch import Tensor

import gmc.mixture as gm
import gmc.mat_tools as mat_tools


def relu(mixture: Tensor, constant: Tensor, n_iter: int = 1) -> typing.Tuple[Tensor, Tensor]:
    weights = gm.weights(mixture)
    positions = gm.positions(mixture)
    covariances = gm.covariances(mixture)
    device = mixture.device

    # todo: make transfer fitting function configurable. e.g. these two can be replaced by leaky relu or softplus (?, might break if we have many overlaying Gs)
    ret_const = constant.where(constant > 0, torch.zeros(1, device=device))
    b = gm.evaluate(mixture, positions) + constant.unsqueeze(-1)
    b = b.where(b > 0, torch.zeros(1, device=device)) - ret_const.unsqueeze(-1)

    x = weights.where(weights + constant.unsqueeze(-1) > 0, -ret_const.unsqueeze(-1))
    x = x.abs() + 0.05

    for i in range(n_iter):
        x = x.abs()
        new_mixture = gm.pack_mixture(x, positions, covariances)
        x = x * b / (gm.evaluate(new_mixture, positions) + 0.01)

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
    exp_values = torch.exp(-0.5 * mat_tools.batched_trace(fitting_covariances.inverse().view(n_batch, n_layers, 1, n_fitting_components, n_dims, n_dims) @
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
    fitting_covariances_inversed = fitting_covariances.inverse().transpose(-2, -1)

    p_diff = target_positions - fitting_positions
    mahalanobis_distance = torch.sqrt(p_diff.unsqueeze(-2) @ fitting_covariances_inversed @ p_diff.unsqueeze(-1)).squeeze(dim=-1).squeeze(dim=-1)
    trace = mat_tools.batched_trace(fitting_covariances_inversed @ target_covariances)
    logarithm = torch.log(target_covariances.det() / fitting_covariances.det())
    KL_divergence = 0.5 * (mahalanobis_distance + trace - 3 - logarithm)

    return KL_divergence


def mhem_algorithm(mixture: Tensor, n_fitting_components: int, n_iterations: int = 1) -> Tensor:
    assert gm.is_valid_mixture(mixture)
    assert n_fitting_components > 0
    n_batch = gm.n_batch(mixture)
    n_layers = gm.n_layers(mixture)
    n_components = gm.n_components(mixture)
    n_dims = gm.n_dimensions(mixture)
    device = mixture.device

    target = mixture
    assert gm.is_valid_mixture(target)

    target_double_gmm, component_integrals, abs_integrals = mixture_to_double_gmm(target)
    # target_double_gmm, integrals = mixture_to_gmm(target)

    # # original
    # _, sorted_indices = torch.sort(component_integrals.detach().abs(), descending=True)
    # fitting_double_gmm = mat_tools.my_index_select(target, sorted_indices[:, :, :n_fitting_components])

    # random sampling
    _, sorted_indices = torch.sort(component_integrals.detach().abs(), descending=True)
    fitting_double_gmm = mat_tools.my_index_select(target, sorted_indices[:, :, :min(max(n_fitting_components*2, n_components // 4), n_components)])
    fitting_double_gmm = mat_tools.my_index_select(fitting_double_gmm, torch.randperm(fitting_double_gmm.shape[-2], device=device)[:n_fitting_components].view(1, 1, n_fitting_components))

    fitting_double_gmm, _, _ = mixture_to_double_gmm(fitting_double_gmm)

    # log(target_double_gmm, 0, tensor_board_writer, layer=layer)
    # log(fitting_double_gmm, 1, tensor_board_writer, layer=layer)

    assert gm.is_valid_mixture(target_double_gmm)
    for i in range(n_iterations):
        assert gm.is_valid_mixture(fitting_double_gmm)

        target_weights = gm.weights(target_double_gmm).unsqueeze(3)
        fitting_weights = gm.weights(fitting_double_gmm).unsqueeze(2)
        sign_match = target_weights.sign() == fitting_weights.sign()
        likelihoods = calc_likelihoods(target_double_gmm.detach(), fitting_double_gmm.detach())
        KL_divergence = calc_KL_divergence(target_double_gmm.detach(), fitting_double_gmm.detach())

        likelihoods = likelihoods * (gm.weights(fitting_double_gmm).abs() / gm.normal_amplitudes(gm.covariances(fitting_double_gmm))).view(n_batch, n_layers, 1, n_fitting_components)
        likelihoods = likelihoods * (KL_divergence < 1) * sign_match

        likelihoods_sum = likelihoods.sum(3, keepdim=True)
        responsibilities = likelihoods / likelihoods_sum.where(likelihoods_sum > 0.0000001, 0.0000001 * torch.ones_like(likelihoods_sum))  # preiner Equation(8)

        assert not torch.any(torch.isnan(responsibilities))

        # index i -> target
        # index s -> fitting
        responsibilities = responsibilities * (gm.weights(target_double_gmm).abs() / gm.normal_amplitudes(gm.covariances(target_double_gmm))).unsqueeze(-1)
        newWeights = torch.sum(responsibilities, 2)
        assert not torch.any(torch.isnan(responsibilities))

        assert not torch.any(torch.isnan(newWeights))
        responsibilities = responsibilities / newWeights.where(newWeights > 0.00000001, 0.00000001 * torch.ones_like(newWeights)).view(n_batch, n_layers, 1, n_fitting_components)
        assert torch.all(responsibilities >= 0)
        assert not torch.any(torch.isnan(responsibilities))
        newPositions = torch.sum(responsibilities.unsqueeze(-1) * gm.positions(target_double_gmm).view(n_batch, n_layers, n_components, 1, n_dims), 2)
        assert not torch.any(torch.isnan(newPositions))
        posDiffs = gm.positions(target_double_gmm).view(n_batch, n_layers, n_components, 1, n_dims, 1) - newPositions.view(n_batch, n_layers, 1, n_fitting_components, n_dims, 1)
        assert not torch.any(torch.isnan(posDiffs))

        newCovariances = (torch.sum(responsibilities.unsqueeze(-1).unsqueeze(-1) * (gm.covariances(target_double_gmm).view(n_batch, n_layers, n_components, 1, n_dims, n_dims) +
                                                                                    posDiffs.matmul(posDiffs.transpose(-1, -2))), 2))
        newCovariances = newCovariances + (newWeights < 0.0001).unsqueeze(-1).unsqueeze(-1) * torch.eye(n_dims, device=device).view(1, 1, 1, n_dims, n_dims) * 0.0001

        assert not torch.any(torch.isnan(newCovariances))

        normal_amplitudes = gm.normal_amplitudes(newCovariances)
        fitting_double_gmm = gm.pack_mixture(newWeights.contiguous() * normal_amplitudes * gm.weights(fitting_double_gmm).sign(), newPositions.contiguous(), newCovariances.contiguous())


    # the following line would have the following effect: scale the fitting result to match the integral of the input exactly. that is bad in case many weak Gs are killed and the remaining weak G is blown up.
    # fitting_double_gmm, _, _, _ = mixture_to_double_gmm(fitting_double_gmm)

    newWeights = gm.weights(fitting_double_gmm) * abs_integrals
    fitting = gm.pack_mixture(newWeights, gm.positions(fitting_double_gmm), gm.covariances(fitting_double_gmm))
    return fitting
