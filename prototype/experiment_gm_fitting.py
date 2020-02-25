import typing
import time
import datetime
import random

import torch
import torch.utils.tensorboard
import torch.distributions.categorical
import torch.optim as optim
import torch.nn as nn
from torch import Tensor

import config
import experiment_gm_mnist
import gm
import gm_fitting

DIMS = 2
N_SAMPLES = 50 * 50
N_INPUT_GAUSSIANS = 10
N_OUTPUT_GAUSSIANS = 10
COVARIANCE_MIN = 0.01

BATCH_SIZE = 200
LEARNING_RATE = 0.001

assert DIMS == 2 or DIMS == 3
assert N_SAMPLES > 0
assert N_INPUT_GAUSSIANS >= N_OUTPUT_GAUSSIANS
assert COVARIANCE_MIN > 0


def generate_random_ReLUandBias(convolved: bool, bias_max: float, weight_min: float, weight_max: float, device: torch.device = 'cpu'):
    # we use the layers for batching so that we can have different biases
    if convolved:
        random_m = gm.generate_random_mixtures(n_batch=1, n_layers=BATCH_SIZE, n_components=random.randint(25, 25), n_dims=DIMS, pos_radius=1, cov_radius=0.10, weight_min=0, weight_max=weight_max, device=device)
        random_kernel = gm.generate_random_mixtures(n_batch=1, n_layers=BATCH_SIZE, n_components=5, n_dims=DIMS, pos_radius=0.1, cov_radius=0.02, device=device)
        weights = gm.weights(random_kernel)
        weights -= weights.mean(dim=2).view(1, -1, 1)
        weights += 0.1
        mixture = gm.convolve(random_m, random_kernel)
    else:
        mixture = gm.generate_random_mixtures(n_batch=1, n_layers=BATCH_SIZE, n_components=N_INPUT_GAUSSIANS, n_dims=DIMS,
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


def generate_simple_fitting_module(n_input_gaussians: int, n_output_gaussians: int) -> gm_fitting.Net:
    assert n_output_gaussians > 0
    n_dimensions = 2
    return gm_fitting.PointNetWithParallelMLPs([64, 128, 128, 512, n_output_gaussians * 25],
                                               [256, 256, 128],
                                               n_output_gaussians=n_output_gaussians,
                                               n_dims=n_dimensions,
                                               aggregations=1, batch_norm=True)

def test_dl_fitting(fitting_function_generator: typing.Callable = generate_simple_fitting_module,
                    device: str = "cuda",
                    epoch_length: int = 100,
                    n_epochs: int = 200,
                    n_input_gaussians=100,
                    n_output_gaussians=25,
                    bias_max: float = 0.65,
                    weight_min: float = -1,
                    weight_max: float = 15,
                    convolved_input: bool = True):
    net = fitting_function_generator(n_input_gaussians, n_output_gaussians)
    net.load()
    net.to(device);

    for parameter in net.parameters():
        print(f"parameter: {parameter.shape}")

    print(net)

    trainer = gm_fitting.Sampler(net, N_SAMPLES, LEARNING_RATE)
    tensor_board_writer = torch.utils.tensorboard.SummaryWriter(config.data_base_path / 'tensorboard' / f'experiment_gm_fitting_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    for i in range(epoch_length * n_epochs):
        mixture, bias = generate_random_ReLUandBias(convolved=convolved_input, bias_max=bias_max, weight_min=weight_min, weight_max=weight_max, device=net.device())
        trainer.run_on(mixture, bias, epoch=i, train=True, tensor_board_writer=tensor_board_writer)
        if i % epoch_length == 0:
            print("testing start")
            net.save()
            trainer.save_optimiser_state()
            experiment_gm_mnist.test_fitting_layer(i/epoch_length,
                                                   tensor_board_writer=tensor_board_writer,
                                                   test_fitting_layers={1},
                                                   layer1_m2m_fitting=fitting_function_generator,
                                                   device=device)
            print("testing end")


test_dl_fitting()
