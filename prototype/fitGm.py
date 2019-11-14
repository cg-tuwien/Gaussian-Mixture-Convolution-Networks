import torch
import torch.distributions.categorical
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import numpy.random as nprnd

import mat_tools
import gm

from gm import ConvolutionLayer
from torch import Tensor


def _select_positions_via_discrete_distribution(layer: ConvolutionLayer, new_n: int) -> Tensor:
    assert new_n < layer.mixture.number_of_components()
    probabilities = layer.evaluate_few_xes(layer.mixture.positions)

    distribution = torch.distributions.categorical.Categorical(probabilities)
    indices = distribution.sample(torch.Size([new_n]))

    return layer.mixture.positions[:, indices]

def _fit_covariances(layer: ConvolutionLayer, positions: Tensor) -> Tensor:
    new_n_components = positions.size()[1]
    covariances = torch.zeros((layer.mixture.covariances.size()[0], new_n_components), dtype=torch.float32, device=layer.device())

    # todo: warning, large matrix (layer.mixture.number_of_components() x new_n_components
    contributions = layer.mixture.evaluate_few_xes_component_wise(positions)

    contributions[contributions < 0] = 0
    weight_sum = contributions.sum(dim=0)
    assert (weight_sum > 0).all()
    contributions /= weight_sum

    for i in range(new_n_components):
        contribution = contributions[:, i]  # old_n_components elements
        weighted_covariances = layer.mixture.covariances[:, i].view(-1, 1) * contribution.view(1, -1)
        covariances[:, i] += weighted_covariances.sum(dim=1)

    return covariances



def test():
    torch.manual_seed(0)
    m1 = gm.Mixture.load("fire_small_mixture")
    # m1.debug_show(-10, -10, 266, 266, 1)

    k1 = gm.generate_null_mixture(9, 2, device=m1.device())
    k1.factors[0] = -1
    k1.factors[1] = 1
    k1.positions[:, 0] = torch.tensor([0, -5], dtype=torch.float32, device=m1.device())
    k1.positions[:, 1] = torch.tensor([0, 5], dtype=torch.float32, device=m1.device())
    k1.covariances[:, 0] = torch.tensor([5, 0, 5], dtype=torch.float32, device=m1.device())
    k1.covariances[:, 1] = torch.tensor([5, 0, 5], dtype=torch.float32, device=m1.device())
    # k1.debug_show(-128, -128, 128, 128, 1)
    conved = gm.convolve(m1, k1)
    layer = ConvolutionLayer(conved, torch.tensor(0.01, dtype=torch.float32, device=m1.device()))
    layer.debug_show(-100, -100, 266, 266, 1)
    positions = _select_positions_via_discrete_distribution(layer, 800)
    weights = layer.evaluate_few_xes(positions)
    covariances = _fit_covariances(layer, positions)
    mc = gm.Mixture(weights, positions, covariances)
    mc.debug_show(-10, -10, 266, 266, 1)

    k2 = gm.generate_random_mixtures(9, 2, device=m1.device())
    # k2.debug_show(-128, -128, 128, 128, 1)
    conved = gm.convolve(m1, k2)
    layer = ConvolutionLayer(conved, torch.tensor(-0.2, dtype=torch.float32, device=m1.device()))
    layer.debug_show(-100, -100, 266, 266, 1)
    positions = _select_positions_via_discrete_distribution(layer, 800)
    weights = layer.evaluate_few_xes(positions)
    covariances = _fit_covariances(layer, positions)
    mc = gm.Mixture(weights, positions, covariances)
    mc.debug_show(-10, -10, 266, 266, 1)

    k3 = gm.generate_random_mixtures(9, 2, device=m1.device())
    k3.debug_show(-128, -128, 128, 128, 1)
    conved = gm.convolve(m1, k3)
    layer = ConvolutionLayer(conved, torch.tensor(-10, dtype=torch.float32, device=m1.device()))
    layer.debug_show(-100, -100, 266, 266, 1.5)
    positions = _select_positions_via_discrete_distribution(layer, 800)
    weights = layer.evaluate_few_xes(positions)
    covariances = _fit_covariances(layer, positions)
    mc = gm.Mixture(weights, positions, covariances)
    mc.debug_show(-10, -10, 266, 266, 1)

test()
