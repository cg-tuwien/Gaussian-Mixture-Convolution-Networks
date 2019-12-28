import typing
import time
import random

import torch
import torch.distributions.categorical
import torch.optim as optim
import torch.nn as nn
from torch import Tensor

import gm
import gm_fitting

DIMS = 2
N_SAMPLES = 50 * 50
N_INPUT_GAUSSIANS = 10
N_OUTPUT_GAUSSIANS = 10
COVARIANCE_MIN = 0.01

BATCH_SIZE = 50
LEARNING_RATE = 0.001

assert DIMS == 2 or DIMS == 3
assert N_SAMPLES > 0
assert N_INPUT_GAUSSIANS >= N_OUTPUT_GAUSSIANS
assert COVARIANCE_MIN > 0


def generate_random_ReLUandBias(convolved: bool, bias_max: float, weight_min: float, weight_max: float, device: torch.device = 'cpu'):
    # we use the layers for batching so that we can have different biases
    if convolved:
        random_m = gm.generate_random_mixtures(1, BATCH_SIZE, random.randint(10, 10), DIMS, pos_radius=1, cov_radius=0.25, weight_min=0, weight_max=weight_max, device=device)
        random_kernel = gm.generate_random_mixtures(1, BATCH_SIZE, 10, DIMS, pos_radius=0.2, cov_radius=0.04, device=device)
        weights = gm.weights(random_kernel)
        weights -= weights.mean(dim=2).view(1, -1, 1)
        weights += 0.1
        mixture = gm.convolve(random_m, random_kernel)
    else:
        mixture = gm.generate_random_mixtures(1, BATCH_SIZE, N_INPUT_GAUSSIANS, DIMS,
                                              pos_radius=1, cov_radius=0.25,
                                              weight_min=weight_min, weight_max=weight_max, device=device)
    # distribution = torch.distributions.categorical.Categorical(torch.ones(N_INPUT_GAUSSIANS, device=device))
    # # zero some input gaussians so we can learn a one to one mapping
    # good_indices = distribution.sample(torch.Size([N_BATCHES, N_OUTPUT_GAUSSIANS]))
    # bool_vector = torch.ones_like(input_gm_after_activation.mixture.weights, dtype=torch.bool)
    # bool_vector[good_indices] = False
    # input_gm_after_activation.mixture.weights[bool_vector] = 0
    # for j in range(input_gm_after_activation.mixture.n_batches()):
    # random_m.debug_show(j, -2, -2, 2, 2, 0.05)
    # random_kernel.debug_show(j, -2, -2, 2, 2, 0.05)
    # input_gm_after_activation.mixture.debug_show(j, -2, -2, 2, 2, 0.05)
    # input_gm_after_activation.debug_show(j, -2, -2, 2, 2, 0.05)
    # print(" ")
    bias = torch.rand(1, BATCH_SIZE, dtype=torch.float32, device=device) * bias_max
    return mixture, bias


def test_dl_fitting(g_layer_sizes: typing.List,
                    fully_layer_sizes: typing.List,
                    device: str = "cuda",
                    testing_mode: bool = False,
                    n_iterations: int = 16385,
                    n_agrs: int = 1,
                    batch_norm: bool = False,
                    bias_max: float = 0.65,
                    weight_min: float = -1,
                    weight_max: float = 15,
                    convolved_input: bool = True):
    net = gm_fitting.Net(g_layer_sizes,
                         fully_layer_sizes,
                         n_output_gaussians=N_OUTPUT_GAUSSIANS,
                         n_dims=DIMS,
                         n_agrs=n_agrs,
                         batch_norm=batch_norm)
    net.load()
    net.to(device);

    for parameter in net.parameters():
        print(f"parameter: {parameter.shape}")

    print(net)

    trainer = gm_fitting.Trainer(net, N_SAMPLES, LEARNING_RATE * float(not testing_mode), not testing_mode, testing_mode)

    for i in range(1 if testing_mode else n_iterations):
        mixture, bias = generate_random_ReLUandBias(convolved=convolved_input, bias_max=bias_max, weight_min=weight_min, weight_max=weight_max, device=net.device())
        trainer.train_on(mixture, bias, i)
        if i % 1000 == 0:
            trainer.save_weights()

    # target, input_ = draw_random_samples(10, WIDTH, HEIGHT)
    # output = net(input_)
    # print(f"target={target}")
    # print(f"output={output}")
    # print(f"diff={output - target}")

