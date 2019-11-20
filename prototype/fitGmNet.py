import pathlib
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

import gm
from gm import MixtureReLUandBias
from gm import Mixture

COVARIANCE_MIN = 0.01


class Net(nn.Module):
    def __init__(self,
                 g_layer_sizes: typing.List,
                 fully_layer_sizes: typing.List,
                 n_input_gaussians: int,
                 n_output_gaussians: int,
                 n_dims: int = 2,
                 cov_decomposition=False):
        super(Net, self).__init__()
        self.n_dims = n_dims
        self.cov_decomposition = cov_decomposition
        self.n_input_gaussians = n_input_gaussians
        self.n_output_gaussians = n_output_gaussians
        # n * (1 for weights, DIMS for positions, trimat_size(DIMS) for the triangle cov matrix) +1 for the bias
        n_inputs_per_gaussian = 1 + n_dims + n_dims * n_dims
        # n_inputs = N_INPUT_GAUSSIANS * (1 + DIMS + mat_tools.trimat_size(DIMS)) + 1
        # and we want to output A, so that C = A @ A.T() + 0.01 * identity() is the cov matrix
        n_outputs_per_gaussian = 1 + n_dims + n_dims * n_dims
        # n_outputs = N_OUTPUT_GAUSSIANS * n_outputs_per_gaussian

        last_layer_size = n_inputs_per_gaussian
        self.per_g_layers = nn.ModuleList()
        for s in g_layer_sizes:
            self.per_g_layers.append(nn.Conv1d(last_layer_size, s, kernel_size=1, stride=1, groups=1))
            last_layer_size = s

        self.fully_layers = nn.ModuleList()
        # todo batching
        # self.batch_norms = nn.ModuleList()

        assert last_layer_size % self.n_output_gaussians == 0
        last_layer_size = last_layer_size // self.n_output_gaussians + 1
        for s in fully_layer_sizes:
            self.fully_layers.append(nn.Conv1d(last_layer_size, s, kernel_size=1, stride=1, groups=1))
            # self.batch_norms.append(nn.BatchNorm1d(s))
            last_layer_size = s

        self.output_layer = nn.Conv1d(last_layer_size, n_outputs_per_gaussian, kernel_size=1, stride=1, groups=1)

        self.name = "fit_gm_net_eigenVecs__g" if self.cov_decomposition else "fit_gm_net__g"
        for s in fully_layer_sizes:
            self.name += f"_{s}"
        self.name += "__f"
        for s in fully_layer_sizes:
            self.name += f"_{s}"

        self.storage_path = "/home/madam/temp/prototype/" + self.name

    def save(self):
        pass
        # torch.save(self.state_dict(), self.storage_path)

    def load(self):
        if pathlib.Path(self.storage_path).is_file():
            state_dict = torch.load(self.storage_path)
            missing_keys, unexpected_keys = self.load_state_dict(state_dict)
            assert len(missing_keys) == 0
            assert len(unexpected_keys) == 0

    def device(self):
        return self.output_layer.bias.device

    def forward(self, convolution_layer: MixtureReLUandBias, learning: bool = True) -> Mixture:
        n_batches = convolution_layer.mixture.n_batches()
        n_input_components = convolution_layer.mixture.n_components()
        n_dims = convolution_layer.mixture.n_dimensions()
        if self.cov_decomposition:
            C = convolution_layer.mixture.covariances
            eigen_vals, eigen_vectors = C.symeig(eigenvectors=True)
            cov_data = eigen_vectors @ torch.sqrt(eigen_vals).diag_embed()
        else:
            cov_data = convolution_layer.mixture.covariances
        x = torch.cat((convolution_layer.mixture.weights.view(n_batches, n_input_components, 1),
                       convolution_layer.mixture.positions,
                       cov_data.view(n_batches, n_input_components, n_dims * n_dims)), dim=2)
        x = x.transpose(1, 2)

        for layer in self.per_g_layers:
            x = layer(x)
            x = F.relu(x)

        x = torch.sum(x, dim=2)
        # x is batch size x final g layer size now
        x = x.view(n_batches, -1, self.n_output_gaussians)
        x = torch.cat((convolution_layer.bias.view(n_batches, 1, 1).expand(n_batches, 1, self.n_output_gaussians), x), dim=1)

        i = 0
        for layer in self.fully_layers:
            x = layer(x)
            x = F.relu(x)
            # x = self.batch_norms[i](x.view(-1, 1))
            i += 1

        x = self.output_layer(x)
        x = x.transpose(1, 2)

        # todo: those magic constants take care of scaling (important for the start). think of something generic, normalisation layer? input normalisation?
        weights = x[:, :, 0]
        positions = x[:, :, 1:(self.n_dims + 1)]
        # we are learning A, so that C = A @ A.T() + 0.01 * identity() is the resulting cov matrix
        A = x[:, :, (self.n_dims + 1):].view(n_batches, -1, self.n_dims, self.n_dims)
        C = A @ A.transpose(2, 3) + torch.eye(self.n_dims, self.n_dims, dtype=torch.float32, device=self.device()) * COVARIANCE_MIN
        covariances = C * 10

        return gm.Mixture(weights, positions, covariances)