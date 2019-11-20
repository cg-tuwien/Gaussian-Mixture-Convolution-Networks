import typing
import time
import pathlib

import torch
import torch.distributions.categorical
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
    k1.weights[0] = -1
    k1.weights[1] = 1
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


def test_dl_fitting(g_layer_sizes: typing.List, fully_layer_sizes: typing.List, use_cuda: bool = True):
    torch.manual_seed(0)
    DIMS = 2
    N_SAMPLES = 50 * 50
    N_INPUT_GAUSSIANS = 10
    N_OUTPUT_GAUSSIANS = 2
    COVARIANCE_MIN = 0.01
    TESTING_MODE = True

    BATCH_SIZE = 200
    LEARNING_RATE = 0.001 / BATCH_SIZE
    N_BATCHES = 5000000

    assert DIMS == 2 or DIMS == 3
    assert N_SAMPLES > 0
    assert N_INPUT_GAUSSIANS >= N_OUTPUT_GAUSSIANS
    assert COVARIANCE_MIN > 0

    class Net(nn.Module):
        def __init__(self, g_layer_sizes: typing.List, fully_layer_sizes: typing.List):
            super(Net, self).__init__()
            # n * (1 for weights, DIMS for positions, trimat_size(DIMS) for the triangle cov matrix) +1 for the bias
            # n_inputs = N_INPUT_GAUSSIANS * (1 + DIMS + mat_tools.trimat_size(DIMS)) + 1
            # and we want to output A, so that C = A @ A.T() + 0.01 * identity() is the cov matrix
            n_outputs_per_gaussian = 1 + DIMS + DIMS * DIMS
            # n_outputs = N_OUTPUT_GAUSSIANS * n_outputs_per_gaussian

            last_layer_size = 6
            self.per_g_layers = nn.ModuleList()
            for s in g_layer_sizes:
                self.per_g_layers.append(nn.Conv1d(last_layer_size, s, kernel_size=1, stride=1, groups=1))
                last_layer_size = s

            self.fully_layers = nn.ModuleList()
            # todo batching
            # self.batch_norms = nn.ModuleList()

            assert last_layer_size % N_OUTPUT_GAUSSIANS == 0
            last_layer_size = last_layer_size // N_OUTPUT_GAUSSIANS + 1
            for s in fully_layer_sizes:
                self.fully_layers.append(nn.Conv1d(last_layer_size, s, kernel_size=1, stride=1, groups=1))
                # self.batch_norms.append(nn.BatchNorm1d(s))
                last_layer_size = s

            self.name = "fit_gm_net__g"
            self.output_layer = nn.Conv1d(last_layer_size, n_outputs_per_gaussian, kernel_size=1, stride=1, groups=1)
            for s in fully_layer_sizes:
                self.name += f"_{s}"
            self.name += "__f"
            for s in fully_layer_sizes:
                self.name += f"_{s}"

            self.storage_path = "/home/madam/temp/prototype/" + self.name

        def save(self):
            assert not TESTING_MODE
            torch.save(self.state_dict(), self.storage_path)

        def load(self):
            if pathlib.Path(self.storage_path).is_file():
                state_dict = torch.load(self.storage_path)
                missing_keys, unexpected_keys = self.load_state_dict(state_dict)
                assert len(missing_keys) == 0
                assert len(unexpected_keys) == 0

        def device(self):
            return self.output_layer.bias.device

        def forward(self, convolution_layer: gm.ConvolutionLayer, learning: bool = True) -> gm.Mixture:
            # todo batching
            batch_size = 1
            x = torch.cat((convolution_layer.mixture.weights.view(1, -1),
                           convolution_layer.mixture.positions,
                           convolution_layer.mixture.covariances), dim=0)
            x = x.reshape(1, 6, -1)

            for layer in self.per_g_layers:
                x = layer(x)
                x = F.relu(x)

            x = torch.sum(x, dim=2)
            # x is batch size x final g layer size now
            x = x.view(batch_size, -1, N_OUTPUT_GAUSSIANS)
            x = torch.cat((convolution_layer.bias.expand(batch_size, 1, N_OUTPUT_GAUSSIANS), x), dim=1)

            i = 0
            for layer in self.fully_layers:
                x = layer(x)
                x = F.relu(x)
                # x = self.batch_norms[i](x.view(-1, 1))
                i += 1

            x = self.output_layer(x)

            # todo: batching, first dimension is the batch (batch_size)
            # todo: those magic constants take care of scaling. think of something generic, normalisation layer? input normalisation?
            weights = x[:, 0, :] * 1
            positions = x[:, 1:(DIMS + 1), :] * 1
            # we are learning A, so that C = A @ A.T() + 0.01 * identity() is the resulting cov matrix
            A = x[:, (DIMS + 1):, :].transpose(1, 2).view(batch_size, -1, DIMS, DIMS)
            C = A @ A.transpose(2, 3) + torch.eye(DIMS, DIMS, dtype=torch.float32, device=self.device()).view(1, 1, DIMS, DIMS) * COVARIANCE_MIN
            # todo: batching, here 0 is the batch id
            covariances = mat_tools.normal_to_triangle(C[0, :, :, :].transpose(0, 2)) * 10

            return gm.Mixture(weights.view(-1), positions.view(2, -1), covariances)

    def generate_random_activation_data():
        if N_INPUT_GAUSSIANS == 100:
            random_m = gm.generate_random_mixtures(10, DIMS, pos_radius=1, cov_radius=0.25, factor_min=0, factor_max=10, device=net.device())
            random_kernel = gm.generate_random_mixtures(10, DIMS, pos_radius=0.08, cov_radius=0.04, device=net.device())
            # todo: print and check factors of convolved gm
            input_gm_after_activation = gm.ConvolutionLayer(gm.convolve(random_m, random_kernel),
                                                            torch.rand(1, dtype=torch.float32, device=net.device()) * 1)
        else:
            input_gm_after_activation = gm.ConvolutionLayer(gm.generate_random_mixtures(N_INPUT_GAUSSIANS, DIMS,
                                                                                        pos_radius=1, cov_radius=0.25,
                                                                                        factor_min=0, factor_max=1, device=net.device()),
                                                            torch.zeros(1, dtype=torch.float32, device=net.device()))
        return input_gm_after_activation

    net = Net(g_layer_sizes, fully_layer_sizes)
    net.load()

    if use_cuda:
        net = net.cuda()
    else:
        net = net.cpu()

    for parameter in net.parameters():
        print(f"parameter: {parameter.shape}")

    criterion = nn.MSELoss()
    optimiser = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    print(net)

    running_loss_avg = 0
    for i in range(1 if TESTING_MODE else N_BATCHES):
        iteration_start_time = time.time()
        optimiser.zero_grad()

        batch_loss_sum = 0
        batch_net_time_sum = 0
        for j in range(BATCH_SIZE):
            input_gm_after_activation = generate_random_activation_data()
            sampling_positions = torch.rand((2, N_SAMPLES), dtype=torch.float32, device=net.device()) * 3 - 1.5
            target_sampling_values = input_gm_after_activation.evaluate_few_xes(sampling_positions)
            # input_gm_after_activation.mixture.debug_show(-1.5, -1.5, 1.5, 1.5, 0.05)
            if TESTING_MODE:
                input_gm_after_activation.debug_show(-2, -2, 2, 2, 0.05)

            network_start_time = time.time()
            output_gm: gm.Mixture = net(input_gm_after_activation)
            batch_net_time_sum += time.time() - network_start_time

            if TESTING_MODE:
                output_gm.debug_show(-2, -2, 2, 2, 0.05)
                input("Press enter to continue")
            output_gm_sampling_values = output_gm.evaluate_few_xes(sampling_positions)

            loss = criterion(output_gm_sampling_values, target_sampling_values)
            loss.backward()
            batch_loss_sum += loss.item()

        if not TESTING_MODE:
            optimiser.step()

        grad_norm_min = 1100000
        grad_norm_sum = 0
        grad_norm_max = 0
        grad_norm_cnt = 0
        running_loss_avg = running_loss_avg * 0.9 + batch_loss_sum * 0.1
        for p in list(filter(lambda p: p.grad is not None, net.parameters())):
            grad_norm = p.grad.data.norm(2).item()
            grad_norm_min = grad_norm if grad_norm < grad_norm_min else grad_norm_min
            grad_norm_max = grad_norm if grad_norm > grad_norm_max else grad_norm_max
            grad_norm_sum += grad_norm
            grad_norm_cnt += 1

        info = (f"iteration i = {i}: "
                f"batch time = {time.time() - iteration_start_time}s (net avg: {batch_net_time_sum / BATCH_SIZE}s,  "
                f"batch loss avg {batch_loss_sum / BATCH_SIZE} (avg10: {running_loss_avg / BATCH_SIZE}, "
                f"grad_norm: {grad_norm_min}/{grad_norm_sum / grad_norm_cnt}/{grad_norm_max}")
        print(info)
        if not TESTING_MODE and i % 10 == 0:
            net.save()
            f = open("/home/madam/temp/prototype/" + net.name + "_loss", "w")
            f.write(info)
            f.close()

    # target, input_ = draw_random_samples(10, WIDTH, HEIGHT)
    # output = net(input_)
    # print(f"target={target}")
    # print(f"output={output}")
    # print(f"diff={output - target}")


# test_dl_fitting([1200]) ## all over the place
# test_dl_fitting([601]) ## all over the place
# test_dl_fitting([200]) ## all over the place

# test_dl_fitting([600, 400]) # averages over all gms? seems to be a blob in the middle very often
# test_dl_fitting([400, 200])

# test_dl_fitting([400, 200, 100]) # there might be some correlation with the height
# test_dl_fitting([600, 400, 200]) # also looks like blob in the middle
#
# test_dl_fitting([600, 400, 200, 100])
#
# test_dl_fitting([600, 200, 200, 200, 100], use_cuda=True)


# test_dl_fitting([61, 61, 61, 30, 30, 14])
# test_dl_fitting([61, 30, 14])
# test_dl_fitting([120, 120, 60, 60, 30, 14])
# test_dl_fitting([600, 600, 400, 200, 100, 30])
# test_dl_fitting([100, 100, 100, 100, 100, 30])

# test_dl_fitting([128, 128], [129, 129, 40])
test_dl_fitting(g_layer_sizes=[256, 256, 256], fully_layer_sizes=[128, 128, 32])
