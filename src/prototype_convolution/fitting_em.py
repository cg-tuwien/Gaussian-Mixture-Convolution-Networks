import time
import typing

import torch
from torch import Tensor

# import update_syspath
import gmc.mixture as gm
import gmc.mat_tools as mat_tools
# from gmc.cpp.extensions.em_fitting.em_fitting import apply as em_fitting

import prototype_convolution.fitting_net as fitting_net

def log(mixture: Tensor, epoch: int, tensor_board_writer, layer: int = 1):
    device = mixture.device
    image_size = 80
    xv, yv = torch.meshgrid([torch.arange(-1.2, 1.2, 2.4 / image_size, dtype=torch.float, device=device),
                             torch.arange(-1.2, 1.2, 2.4 / image_size, dtype=torch.float, device=device)])
    xes = torch.cat((xv.reshape(-1, 1), yv.reshape(-1, 1)), 1).view(1, 1, -1, 2)

    image = gm.evaluate(mixture.detach(), xes).view(-1, image_size, image_size)
    fitting_net.Sampler.log_images(tensor_board_writer,
                                  f"l{layer}_fitting",
                                   [image.transpose(0, 1).reshape(image_size, -1)],
                                   epoch, [-0.5, 2])

def relu(mixture: Tensor, bias: Tensor, n_iter: int = 0) -> Tensor:
    weights = gm.weights(mixture)
    positions = gm.positions(mixture)
    covariances = gm.covariances(mixture)
    bias = bias.unsqueeze(-1)
    device = mixture.device

    b = gm.evaluate(mixture, positions) + bias
    b = b.where(b >= 0, torch.zeros(1, device=device)) - bias      # todo: make transfer fitting function configurable. e.g. these two can be replaced by leaky relu or softplus (?, might break if we have many overlayin Gs)
    ret_bias = bias.where(bias >= 0, torch.zeros(1, device=device))

    inv_eps = 0.0001
    w0 = weights.where((weights < 0) | (weights > inv_eps), torch.ones(1, device=device) * inv_eps)
    w0 = w0.where((weights > 0) | (weights < -inv_eps), - torch.ones(1, device=device) * inv_eps)
    w0inv = 1 / w0

    #x0 = torch.ones_like(w0).where(weights + bias > 0, torch.zeros_like(w0).where(bias < 0, -bias / w0))  # should be the same, since ret_bias is already clamped to [0, inf[
    x = torch.ones_like(w0).where(weights + bias > 0, -ret_bias * w0inv)
    # new_weights = weights.where(weights + bias > 0, -ret_bias)

    for i in range(n_iter):
        new_weights = x * weights
        new_mixture = gm.pack_mixture(new_weights, positions, covariances)
        x = w0inv * (b - gm.evaluate(new_mixture, positions)) - x

        # new_mixture = gm.pack_mixture(new_weights, positions, covariances)
        # new_weights = b - gm.evaluate(new_mixture, positions) - new_weights

    # positive_weights = weights.where(weights > 0, torch.zeros(1, device=device))
    # negative_m = gm.pack_mixture(negative_weights, positions, covariances)
    # positive_m = gm.pack_mixture(positive_weights, positions, covariances)
    # negative_eval = gm.evaluate(negative_m, positions) - bias.unsqueeze(-1)
    # positive_eval = gm.evaluate(positive_m, positions)
    # new_weights_factor = torch.max(torch.zeros(1, device=device),
    #                                torch.ones(1, device=device) + (negative_eval - 0.0001) / (positive_eval + 0.0001))
    # new_weights = new_weights_factor * positive_weights

    new_weights = x * weights

    return gm.pack_mixture(new_weights, positions, covariances), ret_bias

def mixture_to_gmm(mixture: Tensor) -> typing.Tuple[Tensor, Tensor]:
    integrals = gm.integrate(mixture).view(gm.n_batch(mixture), gm.n_layers(mixture), 1)
    integrals = integrals.where(integrals > 0.0001, 0.0001 * torch.ones_like(integrals))
    assert not torch.any(torch.isnan(integrals))

    if torch.any(integrals == 0):
        print("dd")
        assert False

    weights = gm.weights(mixture)
    assert not torch.any(torch.isnan(weights))
    weights = weights / integrals
    assert not torch.any(torch.isnan(weights))
    return gm.pack_mixture(weights, gm.positions(mixture), gm.covariances(mixture)), integrals


def calc_likelihoods(target: Tensor, fitting: Tensor) -> Tensor:
    # index i -> target
    # index s -> fitting
    n_batch = gm.n_batch(target)
    n_layers = gm.n_layers(target)
    n_target_components = gm.n_components(target)
    n_fitting_components = gm.n_components(fitting)
    n_dims = gm.n_dimensions(target)
    n_virtual_points = n_fitting_components

    target_weights = gm.weights(target)
    target_positions = gm.positions(target)
    target_covariances = gm.covariances(target)
    target_normal_amplitudes = gm.normal_amplitudes(target_covariances)

    fitting_weights = gm.weights(fitting)
    fitting_positions = gm.positions(fitting)
    fitting_covariances = gm.covariances(fitting)
    fitting_normal_amplitudes = gm.normal_amplitudes(fitting_covariances)

    target_n_virtual_points = n_virtual_points * target_weights / target_normal_amplitudes;

    # preiner equation 9
    gaussian_values = gm.evaluate_componentwise(gm.pack_mixture(fitting_normal_amplitudes, fitting_positions, fitting_covariances),
                                                     target_positions)
    exp_values = torch.exp(-0.5 * mat_tools.batched_trace(fitting_covariances.inverse().view(n_batch, n_layers, 1, n_fitting_components, n_dims, n_dims) @
                                              target_covariances.view(n_batch, n_layers, n_target_components, 1, n_dims, n_dims)))

    almost_likelihoods = gaussian_values * exp_values

    return torch.pow(almost_likelihoods, target_n_virtual_points.unsqueeze(-1))


def calc_KL_divergence(target: Tensor, fitting: Tensor) -> Tensor:
    # index i -> target
    # index s -> fitting
    n_batch = gm.n_batch(target)
    n_layers = gm.n_layers(target)
    n_target_components = gm.n_components(target)
    n_fitting_components = gm.n_components(fitting)
    n_dims = gm.n_dimensions(target)

    # target_weights = gm.weights(target).unsqueeze(3)
    target_positions = gm.positions(target).unsqueeze(3)
    target_covariances = gm.covariances(target).unsqueeze(3)
    # target_normal_amplitudes = gm.normal_amplitudes(target_covariances)

    # fitting_weights = gm.weights(fitting).unsqueeze(2)
    fitting_positions = gm.positions(fitting).unsqueeze(2)
    fitting_covariances = gm.covariances(fitting).unsqueeze(2)
    fitting_covariances_inversed = fitting_covariances.inverse().transpose(-2, -1)
    # fitting_normal_amplitudes = gm.normal_amplitudes(fitting_covariances)

    p_diff = target_positions - fitting_positions
    mahalanobis_distance = torch.sqrt(p_diff.unsqueeze(-2) @ fitting_covariances_inversed @ p_diff.unsqueeze(-1)).squeeze(dim=-1).squeeze(dim=-1)
    trace = mat_tools.batched_trace(fitting_covariances_inversed @ target_covariances)
    logarithm = torch.log(target_covariances.det() / fitting_covariances.det())
    KL_divergence = 0.5 * (mahalanobis_distance + trace - 3 - logarithm)

    return KL_divergence


def em_algorithm(mixture: Tensor, bias: Tensor, n_fitting_components: int, n_iterations: int = 1, tensor_board_writer = None, layer: int = 0) -> Tensor:
    assert gm.is_valid_mixture(mixture)
    assert n_fitting_components > 0
    n_batch = gm.n_batch(mixture)
    n_layers = gm.n_layers(mixture)
    n_components = gm.n_components(mixture)
    n_dims = gm.n_dimensions(mixture)
    device = mixture.device

    target = relu(mixture, bias)
    assert gm.is_valid_mixture(target)
    component_integrals = gm.integrate_components(target)
    target_gmm, integrals = mixture_to_gmm(target)
    assert gm.is_valid_mixture(target_gmm)

    _, sorted_indices = torch.sort(component_integrals, descending=True)

    fitting_gmm = mat_tools.my_index_select(target, sorted_indices[:, :, :n_fitting_components])
    fitting_gmm, _ = mixture_to_gmm(fitting_gmm)

    # log(target_gmm, 0, tensor_board_writer, layer=layer)
    # log(fitting_gmm, 1, tensor_board_writer, layer=layer)

    fitting_start = time.time()
    assert gm.is_valid_mixture(target_gmm)
    for i in range(n_iterations):
        assert gm.is_valid_mixture(fitting_gmm)
        # likelihoods: Tensor = em_fitting(target_inv, fitting_inv)

        likelihoods = calc_likelihoods(target_gmm.detach(), fitting_gmm.detach())
        KL_divergence = calc_KL_divergence(target_gmm.detach(), fitting_gmm.detach())
        likelihoods = likelihoods * (gm.weights(fitting_gmm) / gm.normal_amplitudes(gm.covariances(fitting_gmm))).view(n_batch, n_layers, 1, n_fitting_components)
        likelihoods = likelihoods * (KL_divergence < 1)
        likelihoods_sum = likelihoods.sum(3, keepdim=True)
        responsibilities = likelihoods / likelihoods_sum.where(likelihoods_sum > 0.00000001, 0.00000001 * torch.ones_like(likelihoods_sum));  # preiner Equation(8)

        assert not torch.any(torch.isnan(responsibilities))

        # index i -> target
        # index s -> fitting
        responsibilities = responsibilities * (gm.weights(target_gmm) / gm.normal_amplitudes(gm.covariances(target_gmm))).unsqueeze(-1);
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
        # normal_amplitudes = normal_amplitudes.where(normal_amplitudes < 10000, 10000 * torch.ones_like(normal_amplitudes))
        fitting_gmm = gm.pack_mixture(newWeights.contiguous() * normal_amplitudes, newPositions.contiguous(), newCovariances.contiguous())
        # log(fitting_gmm, i+1, tensor_board_writer, layer=layer)

    fitting_end = time.time()
    fitting = gm.pack_mixture(gm.weights(fitting_gmm) * integrals, gm.positions(fitting_gmm), gm.covariances(fitting_gmm))
    # print(f"fitting time: {fitting_end - fitting_start}")
    return fitting, responsibilities