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
    DIMS = 2
    N_SAMPLES = 100 * 100
    N_INPUT_GAUSSIANS = 100
    N_OUTPUT_GAUSSIANS = 10
    COVARIANCE_MIN = 0.01

    assert DIMS == 2 or DIMS == 3
    assert N_SAMPLES > 0
    assert N_INPUT_GAUSSIANS >= N_OUTPUT_GAUSSIANS
    assert COVARIANCE_MIN > 0

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            # n * (1 for weights, DIMS for positions, trimat_size(DIMS) for the triangle cov matrix) +1 for the bias
            n_inputs = N_INPUT_GAUSSIANS * (1 + DIMS + mat_tools.trimat_size(DIMS)) + 1
            n_outputs = N_OUTPUT_GAUSSIANS * (1 + DIMS + mat_tools.trimat_size(DIMS))

            self.fc1 = nn.Linear(n_inputs, n_inputs)
            self.fc2 = nn.Linear(n_inputs, n_inputs // 4)
            self.fc3 = nn.Linear(n_inputs // 4, n_inputs // 4)
            self.fc4 = nn.Linear(n_inputs // 4, n_inputs // 4)
            self.fc5 = nn.Linear(n_outputs, n_outputs)

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

            weights = x[0:N_OUTPUT_GAUSSIANS]
            positions = x[N_OUTPUT_GAUSSIANS:3*N_OUTPUT_GAUSSIANS].view(DIMS, -1)
            covariances = x[3*N_OUTPUT_GAUSSIANS:-1].view(mat_tools.trimat_size(DIMS), -1)

            covariance_badness = -1
            if learning:
                if DIMS == 2:
                    indices = (0, 2)
                else:
                    indices = (0, 3, 5)

                bad_variance_indices = covariances[indices, :] < 0
                variance_badness = (covariances[bad_variance_indices] ** 2).sum()
                covariances[bad_variance_indices] = torch.tensor(COVARIANCE_MIN, dtype=torch.float32, device=convolution_layer.device())

                dets = mat_tools.triangle_det(covariances)
                det_badness = ((-dets[dets < COVARIANCE_MIN] + COVARIANCE_MIN) ** 2).sum()

                # todo: how to change cov by adding ids so that the det becomes positive?
                # todo: write the rest of the learning code (gen random ConvolutionLayers, eval for N_SAMPLE positions,
                #  eval the resulting GMs at the same positions, compute square difference, use it together with covariance_badness
                #  for learning

                covariance_badness == variance_badness + det_badness

            return gm.Mixture(weights, positions, covariances, device=convolution_layer.device()), covariance_badness
