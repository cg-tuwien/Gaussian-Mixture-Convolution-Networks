import torch
import torch.distributions.categorical
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import numpy.random as nprnd

import mat_tools
import gm

import fitImage
from gm import Mixture
from gm import ConvolutionLayer
from torch import Tensor


def _select_positions_via_discrete_distribution(layer: ConvolutionLayer, new_n: int) -> Tensor:
    assert new_n < layer.number_of_components()
    probabilities = layer.evaluate_few_xes(layer.mixture.positions)

    probabilities[probabilities < 0] = 0

    distribution = torch.distributions.categorical.Categorical(probabilities)
    indices = distribution.sample(torch.Size([new_n]))

    return layer.positions[:, indices]

def _fit_covariances(layer: ConvolutionLayer, positions: Tensor) -> Tensor:
    weights = layer.mixture.

def test():
    image: np.ndarray = plt.imread("/home/madam/cloud/Photos/fire_small.jpg")
    image = image.mean(axis=2)
    m1 = fitImage.em_algorithm(torch.tensor(image, dtype=torch.float32), n_components=800, n_iterations=0, device='cpu')
    m1.debug_show(0, 0, image.shape[1], image.shape[0], 1.5)

    k1 = gm.generate_null_mixture(9, 2, device=m1.device())
    k1.factors[0] = -1
    k1.factors[1] = 1
    k1.positions[:, 0] = torch.tensor([0, -5], dtype=torch.float32, device=m1.device())
    k1.positions[:, 1] = torch.tensor([0, 5], dtype=torch.float32, device=m1.device())
    k1.covariances[:, 0] = torch.tensor([5, 0, 5], dtype=torch.float32, device=m1.device())
    k1.covariances[:, 1] = torch.tensor([5, 0, 5], dtype=torch.float32, device=m1.device())
    k1.debug_show(-128, -128, 128, 128, 1)

    k2 = gm.generate_random_mixtures(9, 2, device=m1.device())
    k2.debug_show(-128, -128, 128, 128, 1)

    k3 = gm.generate_random_mixtures(9, 2, device=m1.device())
    k3.debug_show(-128, -128, 128, 128, 1)

    conv_start = time.time()
    conved1 = gm.convolve(m1, k1)
    conved2 = gm.convolve(m1, k2)
    conved3 = gm.convolve(m1, k3)
    conv_end = time.time()
    print(f"convolution time: {conv_end - conv_start}")

    conved1.debug_show(0, 0, image.shape[1], image.shape[0], 1.5)
    random_select(conved1, 800).debug_show(0, 0, image.shape[1], image.shape[0], 1.5)
    # conved1.show_after_activation(0, 0, image.shape[1], image.shape[0], 1.5)
    conved2.debug_show(0, 0, image.shape[1], image.shape[0], 1.5)
    random_select(conved2, 800).debug_show(0, 0, image.shape[1], image.shape[0], 1.5)
    # conved2.show_after_activation(0, 0, image.shape[1], image.shape[0], 1.5)
    conved3.debug_show(0, 0, image.shape[1], image.shape[0], 1.5)
    random_select(conved3, 800).debug_show(0, 0, image.shape[1], image.shape[0], 1.5)
    # conved3.show_after_activation(0, 0, image.shape[1], image.shape[0], 1.5)



test()
