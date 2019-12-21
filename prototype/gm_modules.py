import torch
import torch.nn.modules

import gm
import gm_fitting
import mat_tools


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
        # self.kernels = []

        for i in range(self.n_layers_out):
            k = gm.generate_random_mixtures(n_batch=1, n_layers=n_layers_in, n_components=n_kernel_components, n_dims=n_dims,
                                            pos_radius=position_range, cov_radius=covariance_range, weight_min=weight_min, weight_max=weight_max)
            # positive mean produces a rather positive gm. i believe this is a better init
            k.weights -= k.weights.mean(dim=2).view(1, -1, 1) - 0.2
            # self.kernels.append(k)
            self.weights.append(torch.nn.Parameter(k.weights))
            self.positions.append(torch.nn.Parameter(k.positions))
            self.covariances.append(torch.nn.Parameter(k.covariances))

    def forward(self, x: gm.Mixture) -> gm.Mixture:
        out_mixtures = []
        out_mixture_shape = x.shape
        out_mixture_shape[1] = 1
        out_mixture_shape[2] = -1

        for i in range(self.n_layers_out):
            k = gm.pack_mixture(self.weights[i], self.positions[i], self.covariances[i])
            m = gm.convolve(x, k)
            out_mixtures.append(m.view(out_mixture_shape))

        return torch.cat(out_mixtures, dim=1)


class GmBiasAndRelu(torch.nn.modules.Module):
    def __init__(self, n_layers: int, n_output_gaussians: int = 10, n_dimensions=2, train_fitting_net: bool = False):
        # todo support variable number of outputs and configurable net archs. needs a better init + training routine (start with few gaussians etc?)
        assert n_output_gaussians == 10
        super(GmBiasAndRelu, self).__init__()
        self.n_layers = n_layers
        self.n_output_gaussians = n_output_gaussians
        self.train_fitting_net = train_fitting_net
        # use a small bias for the start. i hope it's easier for the net to increase it than to lower it
        self.bias = torch.nn.Parameter(torch.rand(self.n_layers) * 0.2)

        self.net = gm_fitting.Net([64, 128, 256, 512, 1024, 1024, 750],
                                  [256, 256, 256, 256, 256, 128],
                                  n_output_gaussians=n_output_gaussians,
                                  n_dims=n_dimensions,
                                  n_agrs=3, batch_norm=True)
        if not self.net.load(strict=True):
            raise Exception(f"Fitting network {self.net.name} not found.")
        self.net.requires_grad_(False)
        print(self.net)
        # todo: option to make fitting net have common or seperate weights per module

    def forward(self, x: gm.Mixture, division_axis=0) -> gm.Mixture:
        # todo: think of something that would make it possible to do live learning of the fitting network
        if x.n_components() < 134:
            x = gm.MixtureReLUandBias(x, torch.abs(self.bias).view(1, -1))
            return self.net(x)[0]
        else:
            sorted_indices = torch.argsort(x.positions[:, :, :, division_axis])
            sorted_mixture = gm.Mixture(mat_tools.my_index_select(x.weights, sorted_indices),
                                        mat_tools.my_index_select(x.positions, sorted_indices),
                                        mat_tools.my_index_select(x.covariances, sorted_indices))
            # sorted_mixture = gm.Mixture(mat_tools.batched_index_select(x.weights, 2, sorted_indices),
            #                             mat_tools.batched_index_select(x.positions, 2, sorted_indices),
            #                             mat_tools.batched_index_select(x.covariances, 2, sorted_indices))
            fitted_left = self.forward(sorted_mixture.select_components(0, x.n_components() // 2), (division_axis + 1) % x.n_dimensions())
            fitted_right = self.forward(sorted_mixture.select_components(x.n_components() // 2, x.n_components()), (division_axis + 1) % x.n_dimensions())
            return gm.cat((fitted_left, fitted_right), dim=2)
