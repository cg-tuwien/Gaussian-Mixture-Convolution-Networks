import time
import typing

import torch
from torch import Tensor

import update_syspath
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

def mixture_to_inversed_gmm(mixture: Tensor) -> typing.Tuple[Tensor, Tensor]:
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
    return gm.pack_mixture(weights, gm.positions(mixture), gm.covariances(mixture).inverse().contiguous()), integrals


def em_algorithm(mixture: Tensor, n_fitting_components: int, n_iterations: int, tensor_board_writer) -> Tensor:
    # todo test (after moving from Mixture class to Tensor data
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
    target_inv, integrals = mixture_to_inversed_gmm(target)
    assert gm.is_valid_mixture(target)

    _, sorted_indices = torch.sort(component_integrals, descending=True)

    fitting = mat_tools.my_index_select(target, sorted_indices[:, :, :10])
    fitting_inv, _ = mixture_to_inversed_gmm(fitting)
    fitting = gm.pack_mixture(gm.weights(fitting) / _, gm.positions(fitting), gm.covariances(fitting))

    log(fitting, 0, tensor_board_writer)

    fitting_start = time.time()
    assert gm.is_valid_mixture(target)
    for i in range(n_iterations):
        assert gm.is_valid_mixture(fitting_inv)
        likelihoods: Tensor = em_fitting(target_inv, fitting_inv)

        likelihoods_sum = likelihoods.sum(3, keepdim=True)

        responsibilities = likelihoods / likelihoods_sum.where(likelihoods_sum > 0.00001, torch.ones_like(likelihoods_sum));  # preiner Equation(8)

        assert not torch.any(torch.isnan(responsibilities))

        # index i -> target
        # index s -> fitting
        responsibilities = responsibilities * gm.weights(target).unsqueeze(-1);
        assert not torch.any(torch.isnan(responsibilities))

        newWeights = torch.sum(responsibilities, 2);
        assert not torch.any(torch.isnan(newWeights))
        responsibilities = responsibilities / newWeights.where(newWeights > 0.00001, torch.ones_like(newWeights)).view(n_batch, n_layers, 1, n_fitting_components);
        assert not torch.any(torch.isnan(responsibilities))
        newPositions = torch.sum(responsibilities.unsqueeze(-1) * gm.positions(target).view(n_batch, n_layers, n_components, 1, n_dims), 2);
        assert not torch.any(torch.isnan(newPositions))
        posDiffs = gm.positions(target).view(n_batch, n_layers, n_components, 1, n_dims, 1) - newPositions.view(n_batch, n_layers, 1, n_fitting_components, n_dims, 1);
        assert not torch.any(torch.isnan(posDiffs))

        newCovariances = (torch.sum(responsibilities.unsqueeze(-1).unsqueeze(-1) * (gm.covariances(target).view(n_batch, n_layers, n_components, 1, n_dims, n_dims) +
                                                                                    posDiffs.matmul(posDiffs.transpose(-1, -2))), 2))

        assert not torch.any(torch.isnan(newCovariances))
        newCovariances = newCovariances + torch.eye(n_dims)

        fitting = gm.pack_mixture(newWeights.contiguous(), newPositions.contiguous(), newCovariances.contiguous())
        fitting_inv = gm.pack_mixture(newWeights.contiguous(), newPositions.contiguous(), newCovariances.inverse().contiguous())
        log(fitting, i+1, tensor_board_writer)

    fitting_end = time.time()
    print(f"fitting time: {fitting_end - fitting_start}")
    return fitting, responsibilities