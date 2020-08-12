import typing

import torch
from torch import Tensor

# import update_syspath
import gmc.mixture as gm
import gmc.mat_tools as mat_tools


# from gmc.cpp.extensions.em_fitting.em_fitting import apply as em_fitting


def relu(mixture: Tensor, constant: Tensor, n_iter: int = 1) -> typing.Tuple[Tensor, Tensor]:
    weights = gm.weights(mixture)
    positions = gm.positions(mixture)
    covariances = gm.covariances(mixture)
    device = mixture.device

    # todo: make transfer fitting function configurable. e.g. these two can be replaced by leaky relu or softplus (?, might break if we have many overlaying Gs)
    ret_const = constant.where(constant > 0, torch.zeros(1, device=device))
    b = gm.evaluate(mixture, positions) + constant.unsqueeze(-1)
    b = b.where(b > 0, torch.zeros(1, device=device)) - ret_const.unsqueeze(-1)

    # inv_eps = 0.1
    # w0 = weights.where((weights < 0) | (weights > inv_eps), torch.ones(1, device=device) * inv_eps)
    # w0 = w0.where((weights > 0) | (weights < -inv_eps), - torch.ones(1, device=device) * inv_eps)
    # w0inv = 1 / w0

    # x0 = torch.ones_like(w0).where(weights + constant > 0, torch.zeros_like(w0).where(constant < 0, -constant / w0))  # should be the same, since ret_bias is already clamped to [0, inf[
    # x = torch.ones_like(w0).where(weights + constant > 0, -ret_bias * w0inv)
    x = weights.where(weights + constant.unsqueeze(-1) > 0, -ret_const.unsqueeze(-1))
    # sign = x.sign().where(x != 0.0, torch.ones_like(x))
    # x = 0.99 * x + 0.05 * sign
    x = x.abs() + 0.05
    # x = x.where(x > inv_eps, torch.ones_like(x) * inv_eps)
    # x = gm.normal_amplitudes(covariances)
    # alpha = 0

    for i in range(n_iter):
        # jakobi (not working):
        # new_mixture = gm.pack_mixture(x, positions, covariances)
        # x = (1/(1 + alpha)) * (b - gm.evaluate(new_mixture, positions) - x)

        # new_mixture = gm.pack_mixture(torch.ones_like(b), positions, covariances)
        x = x.abs()
        # sign = x.sign().where(x != 0.0, torch.ones_like(x))
        new_mixture = gm.pack_mixture(x, positions, covariances)
        x = x * b / (gm.evaluate(new_mixture, positions) + 0.01)

    # positive_weights = weights.where(weights > 0, torch.zeros(1, device=device))
    # negative_m = gm.pack_mixture(negative_weights, positions, covariances)
    # positive_m = gm.pack_mixture(positive_weights, positions, covariances)
    # negative_eval = gm.evaluate(negative_m, positions) - constant.unsqueeze(-1)
    # positive_eval = gm.evaluate(positive_m, positions)
    # new_weights_factor = torch.max(torch.zeros(1, device=device),
    #                                torch.ones(1, device=device) + (negative_eval - 0.0001) / (positive_eval + 0.0001))
    # new_weights = new_weights_factor * positive_weights

    # new_weights = x * weights

    return gm.pack_mixture(x, positions, covariances), ret_const


def positive_mixture_to_gmm(mixture: Tensor) -> typing.Tuple[Tensor, Tensor]:
    weights = gm.weights(mixture)
    assert ((weights >= 0).all())
    integrals = gm.integrate(mixture).view(gm.n_batch(mixture), gm.n_layers(mixture), 1)
    integrals = integrals.where(integrals > 0.0001, 0.0001 * torch.ones_like(integrals))
    assert not torch.any(torch.isnan(integrals))

    if torch.any(integrals == 0):
        print("dd")
        assert False

    assert not torch.any(torch.isnan(weights))
    weights = weights / integrals
    assert not torch.any(torch.isnan(weights))
    return gm.pack_mixture(weights, gm.positions(mixture), gm.covariances(mixture)), integrals


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

    # fitting weights are not used here, only in mhem_algorithm
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

    _, sorted_indices = torch.sort(component_integrals.detach().abs(), descending=True)
    #
    # fitting_double_gmm = mat_tools.my_index_select(target, sorted_indices[:, :, :min(max(n_fitting_components*2, n_components // 4), n_components)])
    # fitting_double_gmm = mat_tools.my_index_select(fitting_double_gmm, torch.randperm(fitting_double_gmm.shape[-2], device=device)[:n_fitting_components].view(1, 1, n_fitting_components))

    fitting_double_gmm = mat_tools.my_index_select(target, sorted_indices[:, :, :n_fitting_components])
    fitting_double_gmm, _, _ = mixture_to_double_gmm(fitting_double_gmm)

    # log(target_double_gmm, 0, tensor_board_writer, layer=layer)
    # log(fitting_double_gmm, 1, tensor_board_writer, layer=layer)

    assert gm.is_valid_mixture(target_double_gmm)
    for i in range(n_iterations):
        assert gm.is_valid_mixture(fitting_double_gmm)
        # likelihoods: Tensor = em_fitting(target_inv, fitting_inv)

        target_weights = gm.weights(target_double_gmm).unsqueeze(3)
        fitting_weights = gm.weights(fitting_double_gmm).unsqueeze(2)
        sign_match = target_weights.sign() == fitting_weights.sign()
        sign_positive = (target_weights.sign() == 1) & (fitting_weights.sign() == 1)
        sign_negative = (target_weights.sign() == -1) & (fitting_weights.sign() == -1)
        likelihoods = calc_likelihoods(target_double_gmm.detach(), fitting_double_gmm.detach())
        KL_divergence = calc_KL_divergence(target_double_gmm.detach(), fitting_double_gmm.detach())

        likelihoods = likelihoods * (gm.weights(fitting_double_gmm).abs() / gm.normal_amplitudes(gm.covariances(fitting_double_gmm))).view(n_batch, n_layers, 1, n_fitting_components)
        likelihoods = likelihoods * (KL_divergence < 1) * sign_match
        likelihoods_positive = likelihoods * sign_positive
        likelihoods_negative = likelihoods * sign_negative
        likelihoods_positive_sum = likelihoods_positive.sum(3, keepdim=True)
        likelihoods_negative_sum = likelihoods_negative.sum(3, keepdim=True)
        responsibilities_positive = likelihoods_positive / likelihoods_positive_sum.where(likelihoods_positive_sum > 0.0000001, 0.0000001 * torch.ones_like(likelihoods_positive_sum))  # preiner Equation(8)
        responsibilities_negative = likelihoods_negative / likelihoods_negative_sum.where(likelihoods_negative_sum > 0.0000001, 0.0000001 * torch.ones_like(likelihoods_negative_sum))  # preiner Equation(8)

        # todo: result looked the same. maybe we can merge back into one walk through?
        def comp_m(responsibilities, fitting_gmm, target_gmm, n_batch, n_layers, n_components, n_dims, n_fitting_components, sign):
            assert not torch.any(torch.isnan(responsibilities))

            # index i -> target
            # index s -> fitting
            responsibilities = responsibilities * (gm.weights(target_gmm).abs() / gm.normal_amplitudes(gm.covariances(target_gmm))).unsqueeze(-1)
            newWeights = torch.sum(responsibilities, 2)
            assert not torch.any(torch.isnan(responsibilities))

            assert not torch.any(torch.isnan(newWeights))
            responsibilities = responsibilities / newWeights.where(newWeights > 0.00000001, 0.00000001 * torch.ones_like(newWeights)).view(n_batch, n_layers, 1, n_fitting_components)
            assert torch.all(responsibilities >= 0)
            assert not torch.any(torch.isnan(responsibilities))
            newPositions = torch.sum(responsibilities.unsqueeze(-1) * gm.positions(target_gmm).view(n_batch, n_layers, n_components, 1, n_dims), 2)
            assert not torch.any(torch.isnan(newPositions))
            posDiffs = gm.positions(target_gmm).view(n_batch, n_layers, n_components, 1, n_dims, 1) - newPositions.view(n_batch, n_layers, 1, n_fitting_components, n_dims, 1)
            assert not torch.any(torch.isnan(posDiffs))

            newCovariances = (torch.sum(responsibilities.unsqueeze(-1).unsqueeze(-1) * (gm.covariances(target_gmm).view(n_batch, n_layers, n_components, 1, n_dims, n_dims) +
                                                                                        posDiffs.matmul(posDiffs.transpose(-1, -2))), 2))
            newCovariances = newCovariances + (newWeights < 0.0001).unsqueeze(-1).unsqueeze(-1) * torch.eye(n_dims, device=device).view(1, 1, 1, n_dims, n_dims) * 0.0001

            assert not torch.any(torch.isnan(newCovariances))

            normal_amplitudes = gm.normal_amplitudes(newCovariances)
            return gm.pack_mixture(newWeights.contiguous() * normal_amplitudes * sign, newPositions.contiguous(), newCovariances.contiguous())

        fitting_double_gmm = comp_m(responsibilities_positive, fitting_double_gmm, target_double_gmm, n_batch, n_layers, n_components, n_dims, n_fitting_components, 1) + \
                             comp_m(responsibilities_negative, fitting_double_gmm, target_double_gmm, n_batch, n_layers, n_components, n_dims, n_fitting_components, -1)

    # the following line would have the following effect: scale the fitting result to match the integral of the input exactly. that is bad in case many weak Gs are killed and the remaining weak G is blown up.
    # fitting_double_gmm, _, _, _ = mixture_to_double_gmm(fitting_double_gmm)

    newWeights = gm.weights(fitting_double_gmm) * abs_integrals
    fitting = gm.pack_mixture(newWeights, gm.positions(fitting_double_gmm), gm.covariances(fitting_double_gmm))
    return fitting
