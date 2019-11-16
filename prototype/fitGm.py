import torch
import torch.distributions.categorical
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

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


def test_manual_heuristic():
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


def test_dl_fitting():
    torch.manual_seed(0)
    DIMS = 2
    N_SAMPLES = 100 * 100
    N_INPUT_GAUSSIANS = 100
    N_OUTPUT_GAUSSIANS = 10
    COVARIANCE_MIN = 0.01

    N_INNER_GRADIENT_STEPS = 10

    assert DIMS == 2 or DIMS == 3
    assert N_SAMPLES > 0
    assert N_INPUT_GAUSSIANS >= N_OUTPUT_GAUSSIANS
    assert COVARIANCE_MIN > 0

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # n * (1 for weights, DIMS for positions, trimat_size(DIMS) for the triangle cov matrix) +1 for the bias
            n_inputs = N_INPUT_GAUSSIANS * (1 + DIMS + mat_tools.trimat_size(DIMS)) + 1
            # and we want to output A, so that C = A @ A.T() + 0.01 * identity() is the cov matrix
            n_outputs = N_OUTPUT_GAUSSIANS * (1 + DIMS + DIMS * DIMS)

            self.fc1 = nn.Linear(n_inputs, n_inputs)
            self.fc2 = nn.Linear(n_inputs, n_inputs // 2)
            self.fc3 = nn.Linear(n_inputs // 2, n_inputs // 4)
            self.fc4 = nn.Linear(n_inputs // 4, n_inputs // 4)
            self.fc5 = nn.Linear(n_inputs // 4, n_outputs)

        def device(self):
            return self.fc1.bias.device

        def forward(self, convolution_layer: gm.ConvolutionLayer, learning: bool = True) -> gm.Mixture:
            x = torch.cat((convolution_layer.bias,
                           convolution_layer.mixture.factors,
                           convolution_layer.mixture.positions.view(-1),
                           convolution_layer.mixture.covariances.view(-1)))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = self.fc5(x)

            weights = x[0:N_OUTPUT_GAUSSIANS] * 10
            positions = x[N_OUTPUT_GAUSSIANS:3*N_OUTPUT_GAUSSIANS].view(DIMS, -1) * 10
            # we are learning A, so that C = A @ A.T() + 0.01 * identity() is the resulting cov matrix
            A = x[3*N_OUTPUT_GAUSSIANS:].view(-1, DIMS, DIMS)
            C = A @ A.transpose(1, 2) + torch.eye(DIMS, DIMS).view(-1, DIMS, DIMS) * COVARIANCE_MIN
            covariances = mat_tools.normal_to_triangle(C.transpose(0, 2)) * 25
            #
            # covariance_badness = -1
            # if learning:
            #     if DIMS == 2:
            #         indices = (0, 2)
            #     else:
            #         indices = (0, 3, 5)
            #
            #     bad_variance_indices = covariances[indices, :] < 0
            #     variance_badness = (covariances[bad_variance_indices] ** 2).sum()
            #     covariances[bad_variance_indices] = torch.tensor(COVARIANCE_MIN, dtype=torch.float32, device=convolution_layer.device())
            #
            #     dets = mat_tools.triangle_det(covariances)
            #     det_badness = ((-dets[dets < COVARIANCE_MIN] + COVARIANCE_MIN) ** 2).sum()
            #
            #     # todo: how to change cov by adding ids so that the det becomes positive?
                # todo: write the rest of the learning code (gen random ConvolutionLayers, eval for N_SAMPLE positions,
                #  eval the resulting GMs at the same positions, compute square difference, use it together with covariance_badness
                #  for learning

            #     covariance_badness == variance_badness + det_badness

            return gm.Mixture(weights, positions, covariances)

    net = Net().cuda()
    for parameter in net.parameters():
        print(f"parameter: {parameter.shape}")

    criterion = nn.MSELoss()
    optimiser = optim.Adam(net.parameters(), lr=0.001)
    print(net)

    for i in range(2000):
        assert N_INPUT_GAUSSIANS == 100
        iteration_start_time = time.time()
        random_m = gm.generate_random_mixtures(10, DIMS, pos_radius=1, cov_radius=0.25, factor_min=0, factor_max=10, device=net.device())
        random_kernel = gm.generate_random_mixtures(10, DIMS, pos_radius=0.08, cov_radius=0.04, device=net.device())
        # todo: print and check factors of convolved gm
        input_gm_after_activation = gm.ConvolutionLayer(gm.convolve(random_m, random_kernel),
                                                        torch.rand(1, dtype=torch.float32, device=net.device())*1)

        sampling_positions = torch.rand((2, N_SAMPLES), dtype=torch.float32, device=net.device()) * 3 - 1.5
        target_sampling_values = input_gm_after_activation.evaluate_few_xes(sampling_positions)
        # input_gm_after_activation.mixture.debug_show(-1.5, -1.5, 1.5, 1.5, 0.05)
        input_gm_after_activation.debug_show(-1.5, -1.5, 1.5, 1.5, 0.05)

        begin_loss = 0
        for j in range(N_INNER_GRADIENT_STEPS):
            inner_start_time = time.time()
            optimiser.zero_grad()
            output_gm: gm.Mixture = net(input_gm_after_activation)
            inner_network_time = time.time()
            output_gm_sampling_values = output_gm.evaluate_few_xes(sampling_positions)

            loss = criterion(output_gm_sampling_values, target_sampling_values)
            loss.backward()
            optimiser.step()
            if j == 0:
                begin_loss = loss
                output_gm.debug_show(-1.5, -1.5, 1.5, 1.5, 0.05)
            print(f"j = {j}: time: {time.time() - inner_start_time}s, of which network: {inner_network_time - inner_start_time}, loss: {loss}")

        # output_gm.debug_show(-1.5, -1.5, 1.5, 1.5, 0.05)
        print(f"iteration i = {i}: time = {time.time() - iteration_start_time},  loss_0 {begin_loss},  loss_{N_INNER_GRADIENT_STEPS} {loss}")

    target, input_ = draw_random_samples(10, WIDTH, HEIGHT)
    output = net(input_)
    print(f"target={target}")
    print(f"output={output}")
    print(f"diff={output - target}")

test_dl_fitting()
