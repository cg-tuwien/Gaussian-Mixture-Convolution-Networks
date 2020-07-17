import time
import typing

import torch
from torch import Tensor

# import update_syspath
import gmc.mixture as gm
import gmc.mat_tools as mat_tools
from gmc.cpp.extensions.em_fitting.em_fitting import apply as em_fitting

import prototype_convolution.fitting_net as fitting_net

def log(mixture: Tensor, epoch: int, tensor_board_writer):
    device = mixture.device
    image_size = 80
    xv, yv = torch.meshgrid([torch.arange(-1.2, 1.2, 2.4 / image_size, dtype=torch.float, device=device),
                             torch.arange(-1.2, 1.2, 2.4 / image_size, dtype=torch.float, device=device)])
    xes = torch.cat((xv.reshape(-1, 1), yv.reshape(-1, 1)), 1).view(1, 1, -1, 2)

    image = gm.evaluate(mixture.detach(), xes).view(-1, image_size, image_size)
    fitting_net.Sampler.log_images(tensor_board_writer,
                                  f"fitting",
                                   [image.transpose(0, 1).reshape(image_size, -1)],
                                   epoch, [-0.5, 2])

def relu(mixture: Tensor) -> Tensor:
    weights = gm.weights(mixture)
    positions = gm.positions(mixture)
    covariances = gm.covariances(mixture)
    device = mixture.device

    negative_weights = weights.where(weights <= 0, torch.zeros(1, device=device))
    positive_weights = weights.where(weights > 0, torch.zeros(1, device=device))
    negative_m = gm.pack_mixture(negative_weights, positions, covariances)
    positive_m = gm.pack_mixture(positive_weights, positions, covariances)
    negative_eval = gm.evaluate(negative_m, positions)
    positive_eval = gm.evaluate(positive_m, positions)
    new_weights_factor = torch.max(torch.zeros(1, device=device),
                                   torch.ones(1, device=device) + (negative_eval - 0.0001) / (positive_eval + 0.0001))
    new_weights = new_weights_factor * positive_weights

    return gm.pack_mixture(new_weights, positions, covariances)

def mixture_to_gmm(mixture: Tensor) -> typing.Tuple[Tensor, Tensor]:
    integrals = gm.integrate(mixture).view(gm.n_batch(mixture), gm.n_layers(mixture), 1)
    integrals = integrals.where(integrals > 0.00001, torch.ones_like(integrals))
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



def em_algorithm(mixture: Tensor, n_fitting_components: int, n_iterations: int, tensor_board_writer) -> Tensor:
    assert gm.is_valid_mixture(mixture)
    assert n_fitting_components > 0
    n_batch = gm.n_batch(mixture)
    n_layers = gm.n_layers(mixture)
    n_components = gm.n_components(mixture)
    n_dims = gm.n_dimensions(mixture)
    device = mixture.device

    target = relu(mixture)
    assert gm.is_valid_mixture(target)
    component_integrals = gm.integrate_components(target)
    target_gmm, integrals = mixture_to_gmm(target)
    assert gm.is_valid_mixture(target_gmm)

    _, sorted_indices = torch.sort(component_integrals, descending=True)

    fitting_gmm = mat_tools.my_index_select(target, sorted_indices[:, :, :n_fitting_components])
    fitting_gmm, _ = mixture_to_gmm(fitting_gmm)

    log(target_gmm, 0, tensor_board_writer)
    log(fitting_gmm, 1, tensor_board_writer)

    fitting_start = time.time()
    assert gm.is_valid_mixture(target_gmm)
    for i in range(n_iterations):
        assert gm.is_valid_mixture(fitting_gmm)
        # likelihoods: Tensor = em_fitting(target_inv, fitting_inv)

        likelihoods = calc_likelihoods(target_gmm, fitting_gmm)
        likelihoods = likelihoods * (gm.weights(fitting_gmm) / gm.normal_amplitudes(gm.covariances(fitting_gmm))).view(n_batch, n_layers, 1, n_fitting_components)
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
        newCovariances = newCovariances + (newWeights == 0).unsqueeze(-1).unsqueeze(-1) * torch.eye(n_dims).view(1, 1, 1, n_dims, n_dims)

        assert not torch.any(torch.isnan(newCovariances))

        fitting_gmm = gm.pack_mixture(newWeights.contiguous() * gm.normal_amplitudes(newCovariances), newPositions.contiguous(), newCovariances.contiguous())
        log(fitting_gmm, i+2, tensor_board_writer)

    fitting_end = time.time()
    fitting = gm.pack_mixture(gm.weights(fitting_gmm) * integrals, gm.positions(fitting_gmm), gm.covariances(fitting_gmm))
    print(f"fitting time: {fitting_end - fitting_start}")
    return fitting, responsibilities