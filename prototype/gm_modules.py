import torch
import torch.nn.modules

import gm


class GmConvolution(torch.nn.modules.Module):
    def __init__(self, n_layers_in: int, n_layers_out: int, n_kernel_components: int = 4, n_dims: int = 2, position_range: float = 1, covariance_range: float = 0.25, weight_min=-1, weight_max=1):
        super(GmConvolution, self).__init__()
        self.n_layers_in = n_layers_in
        self.n_layers_out = n_layers_out
        self.n_kernel_components = n_kernel_components
        self.n_dims = n_dims
        self.position_range = position_range
        self.covariance_range = covariance_range
        self.weight_min = weight_min
        self.weight_max = weight_max

        self.weights = torch.nn.modules.ParameterList()
        self.positions = torch.nn.modules.ParameterList()
        self.covariances = torch.nn.modules.ParameterList()
        self.kernels = []

        for i in range(self.n_layers_out):
            k = gm.generate_random_mixtures(n_layers_in, n_kernel_components, n_dims, pos_radius=position_range, cov_radius=covariance_range, weight_min=weight_min, weight_max=weight_max)
            self.kernels.append(k)
            self.weights.append(torch.nn.Parameter(k.weights))
            self.positions.append(torch.nn.Parameter(k.positions))
            self.covariances.append(torch.nn.Parameter(k.covariances))

    def forward(self, x: gm.Mixture) -> gm.Mixture:
        out_mixtures = []

        for i in range(self.n_layers_out):
            m = gm.convolve(x, self.kernels[i])
            out_mixtures.append(m)

        return gm.batch_sum(out_mixtures)


class GmBiasAndRelu(torch.nn.modules.Module):
    def __init__(self, n_layers: int, n_output_gaussians: int, train_fitting_net: bool = True):
        super(GmBiasAndRelu, self).__init__()
        self.n_layers = n_layers
        self.n_output_gaussians = n_output_gaussians
        self.train_fitting_net = train_fitting_net
        self.bias = torch.nn.Parameter(torch.rand(self.n_layers))

        #todo: load fitting net

        #todo: option to make fitting net have common or seperate weights per module

    def forward(self, x: gm.Mixture) -> gm.Mixture:
        x = gm.MixtureReLUandBias(x, self.bias)


        # todo: think of something that would make it possible to do live learning of the fitting network

        # todo: fitting and return

