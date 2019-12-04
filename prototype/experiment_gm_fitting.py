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
N_INPUT_GAUSSIANS = 600
N_OUTPUT_GAUSSIANS = 10
COVARIANCE_MIN = 0.01

BATCH_SIZE = 50
LEARNING_RATE = 0.001

assert DIMS == 2 or DIMS == 3
assert N_SAMPLES > 0
assert N_INPUT_GAUSSIANS >= N_OUTPUT_GAUSSIANS
assert COVARIANCE_MIN > 0


def generate_random_ReLUandBias(bias_mul: float, weight_min: float, weight_max: float, device: torch.device = 'cpu'):
    random_m = gm.generate_random_mixtures(BATCH_SIZE, random.randint(10, 10), DIMS, pos_radius=1, cov_radius=0.25, weight_min=weight_min, weight_max=weight_max, device=device)
    random_kernel = gm.generate_random_mixtures(BATCH_SIZE, 10, DIMS, pos_radius=0.2, cov_radius=0.04, device=device)
    random_kernel.weights -= random_kernel.weights.mean(dim=1).view(-1, 1)
    random_kernel.weights += 0.1
    # todo: print and check factors of convolved gm
    input_gm_after_activation = gm.MixtureReLUandBias(gm.convolve(random_m, random_kernel),
                                                      torch.rand(BATCH_SIZE, dtype=torch.float32, device=device) * bias_mul)

    # input_gm_after_activation = gm.MixtureReLUandBias(gm.generate_random_mixtures(BATCH_SIZE, N_INPUT_GAUSSIANS, DIMS,
    #                                                                               pos_radius=1, cov_radius=0.25,
    #                                                                               factor_min=weight_min, factor_max=weight_max, device=device),
    #                                                   torch.rand(BATCH_SIZE, dtype=torch.float32, device=device) * bias_mul)
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
    return input_gm_after_activation


def test_dl_fitting(g_layer_sizes: typing.List,
                    fully_layer_sizes: typing.List,
                    use_cuda: bool = True,
                    testing_mode: bool = True,
                    n_iterations: int = 50001,
                    bias_mul: float = 1,
                    weight_min: float = -1,
                    weight_max: float = 1):
    net = gm_fitting.Net(g_layer_sizes,
                         fully_layer_sizes,
                         n_output_gaussians=N_OUTPUT_GAUSSIANS,
                         n_dims=DIMS,
                         output_image_width = 128,
                         output_image_height = 128)
    net.load()

    if use_cuda:
        net = net.cuda()
    else:
        net = net.cpu()

    for parameter in net.parameters():
        print(f"parameter: {parameter.shape}")

    print(net)

    trainer = gm_fitting.Trainer(net, N_SAMPLES, LEARNING_RATE * float(not testing_mode), not testing_mode, testing_mode)

    for i in range(1 if testing_mode else n_iterations):
        input_relu_of_gm_p_bias = generate_random_ReLUandBias(bias_mul=bias_mul, weight_min=weight_min, weight_max=weight_max, device=net.device())
        trainer.save_weights = i % 50 == 0
        trainer.train_on(input_relu_of_gm_p_bias, i)

    # target, input_ = draw_random_samples(10, WIDTH, HEIGHT)
    # output = net(input_)
    # print(f"target={target}")
    # print(f"output={output}")
    # print(f"diff={output - target}")


# test_dl_fitting(g_layer_sizes=[64, 128, 128, 512, 512 * N_OUTPUT_GAUSSIANS], fully_layer_sizes=[512, 256, 128, 64, 32],
#                 use_cuda=True, cov_decomposition=False, testing_mode=False, bias_mul=0, weight_min=0)
#
# test_dl_fitting(g_layer_sizes=[64, 64, 128, 128, 512, 512 * N_OUTPUT_GAUSSIANS], fully_layer_sizes=[512, 256, 128, 64, 32],
#                 use_cuda=True, cov_decomposition=False, testing_mode=False, bias_mul=0, weight_min=0)
#
# test_dl_fitting(g_layer_sizes=[64, 64, 128, 128, 512, 1024 * N_OUTPUT_GAUSSIANS], fully_layer_sizes=[512, 256, 128, 64, 32],
#                 use_cuda=True, cov_decomposition=False, testing_mode=False, bias_mul=0, weight_min=0)
#

test_dl_fitting(g_layer_sizes=[64, 128, 256, 512, 5000], fully_layer_sizes=[128, 128, 64, 32], testing_mode=False, bias_mul=0.65, weight_min=0, weight_max=15)

# test_dl_fitting(g_layer_sizes=[64, 64, 128, 128, 512, 1024 * N_OUTPUT_GAUSSIANS], fully_layer_sizes=[512, 256, 128, 64, 32],
#                 use_cuda=True, cov_decomposition=False, testing_mode=False, bias_mul=1, weight_min=-1)
#
# test_dl_fitting(g_layer_sizes=[64, 128, 128, 512, 512 * N_OUTPUT_GAUSSIANS], fully_layer_sizes=[512, 256, 128, 64, 32],
#                 use_cuda=False, cov_decomposition=True, testing_mode=False, bias_mul=0, weight_min=0)


# test_dl_fitting(g_layer_sizes=[32, 64, 128, 128, 256, 256, 512, 1024 * N_OUTPUT_GAUSSIANS], fully_layer_sizes=[1024, 512, 256, 256, 128, 128, 64, 32],
#                 use_cuda=True, cov_decomposition=False, testing_mode=False, bias_mul=1, weight_min=-4, weight_max=4)

# def test_dl_fitting(g_layer_sizes: typing.List,
#                     fully_layer_sizes: typing.List,
#                     use_cuda: bool = True,
#                     cov_decomposition: bool = True,
#                     testing_mode: bool = True,
#                     n_iterations: int = 50000,
#                     bias_mul: float = 1,
#                     weight_min: float = -1):
#     pass
# test_dl_fitting(g_layer_sizes=[64, 128, 128, 512, 512 * N_OUTPUT_GAUSSIANS], fully_layer_sizes=[512, 256, 128, 64, 32])
