import time
import typing

import torch
from torch import Tensor

import update_syspath
import gmc.mixture as gm
from gmc.cpp.extensions.em_fitting.em_fitting import apply as em_fitting


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


def em_algorithm(mixture: Tensor, n_components: int, n_iterations: int) -> Tensor:
    # todo test (after moving from Mixture class to Tensor data
    assert gm.is_valid_mixture(mixture)
    assert n_components > 0
    n_batch = gm.n_batch(mixture)
    n_layers = gm.n_layers(mixture)
    n_dims = gm.n_dimensions(mixture)
    device = mixture.device

    target = relu(mixture)
    assert gm.is_valid_mixture(target)
    target, integrals = mixture_to_inversed_gmm(target)
    assert gm.is_valid_mixture(target)
    initial_fitting, _ = mixture_to_inversed_gmm(gm.generate_random_mixtures(n_batch, n_layers, n_components, n_dims, device=device))

    fitting_start = time.time()
    assert gm.is_valid_mixture(target)
    assert gm.is_valid_mixture(initial_fitting)
    fitting = em_fitting(target, initial_fitting)

    fitting_end = time.time()
    print(f"fitting time: {fitting_end - fitting_start}")
    return fitting