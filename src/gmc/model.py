from __future__ import print_function
import time
import typing

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter as TensorboardWriter

import gmc.mixture as gm
import gmc.modules
import gmc.mat_tools as mat_tools


class Layer:
    def __init__(self, n_feature_maps, kernel_radius, n_fitting_components):
        self.n_feature_layers = n_feature_maps
        self.kernel_radius = kernel_radius
        self.n_fitting_components = n_fitting_components


class Config:
    BN_CONSTANT_COMPUTATION_ZERO = 0
    BN_CONSTANT_COMPUTATION_MEAN_IN_CONST = 1
    BN_CONSTANT_COMPUTATION_INTEGRAL = 2
    BN_CONSTANT_COMPUTATION_WEIGHTED = 3

    BIAS_TYPE_NONE = 0
    BIAS_TYPE_NORMAL = 1
    BIAS_TYPE_NEGATIVE_SOFTPLUS = 2

    BN_TYPE_STD = "Std"
    BN_TYPE_COVARIANCE = "Cov"
    BN_TYPE_COVARIANCE_STD = "CovStd"

    BN_PLACE_NOWHERE = "None"
    BN_PLACE_AFTER_GMC = "aCn"
    BN_PLACE_AFTER_RELU = "aRl"

    def __init__(self, n_dims):
        self.n_dims = n_dims
        self.n_classes = 10

        # complexity / power / number of parameters
        self.n_kernel_components = 5
        self.layers: typing.List[Layer] = [Layer(8, 1, 32),
                                           Layer(16, 1, 16),
                                           Layer(10, 1, -1)]
        self.bias_type = Config.BIAS_TYPE_NONE
        self.mlp: typing.Optional[typing.List[int]] = None

        # auxiliary architectural options
        self.bn_place = Config.BN_PLACE_AFTER_RELU
        self.bn_type = Config.BN_TYPE_COVARIANCE_STD
        self.dropout = 0.0

        self.relu_config: gmc.modules.ReLUFittingConfig = gmc.modules.ReLUFittingConfig()
        self.convolution_config: gmc.modules.ConvolutionConfig = gmc.modules.ConvolutionConfig()

    def produce_gmc_layers_description(self) -> str:
        name = "L"
        for l in self.layers:
            name = f"{name}_{l.n_feature_layers}f_{int(l.kernel_radius * 10)}r_{int(l.n_fitting_components)}c"
        return name

    def produce_description(self):
        mlp_string = ""
        if self.mlp is not None:
            mlp_string = "_MLP"
            for l in self.mlp:
                mlp_string = f"{mlp_string}_{l}"
        return f"BN{self.bn_place}{self.bn_type}_Drp{int(self.dropout * 100)}" \
               f"_nK{self.n_kernel_components}_{self.produce_gmc_layers_description()}{mlp_string}_"


class Net(nn.Module):
    def __init__(self,
                 learn_positions: bool = True,
                 learn_covariances: bool = True,
                 config: Config = Config):
        super(Net, self).__init__()

        self.config = config

        bias_0 = 0.0
        if self.config.bias_type == Config.BIAS_TYPE_NEGATIVE_SOFTPLUS:
            bias_0 = -0.1

        def pos2cov(p): return (p / 3) ** 2

        self.norm0 = nn.Sequential(gmc.modules.CovScaleNorm(1, batch_norm=False),
                                   gmc.modules.BatchNorm(1, batch_norm=False))
        self.biases = torch.nn.ParameterList()
        self.gmcs = torch.nn.modules.ModuleList()
        self.relus = torch.nn.modules.ModuleList()
        self.norms = torch.nn.modules.ModuleList()
        self.dropout = gmc.modules.Dropout(config.dropout)

        n_feature_channels_in = 1
        for i, l in enumerate(config.layers):
            self.gmcs.append(gmc.modules.Convolution(config.convolution_config, n_layers_in=n_feature_channels_in, n_layers_out=l.n_feature_layers, n_kernel_components=config.n_kernel_components,
                                                     position_range=l.kernel_radius, covariance_range=pos2cov(l.kernel_radius),
                                                     learn_positions=learn_positions, learn_covariances=learn_covariances,
                                                     weight_sd=1, weight_mean=0.1, n_dims=config.n_dims))
            self.biases.append(torch.nn.Parameter(torch.zeros(1, l.n_feature_layers) + bias_0))
            self.relus.append(gmc.modules.ReLUFitting(config.relu_config, n_layers=l.n_feature_layers, n_output_gaussians=l.n_fitting_components))
            if config.bn_type == Config.BN_TYPE_COVARIANCE_STD:
                norm = nn.Sequential(gmc.modules.CovScaleNorm(n_layers=l.n_feature_layers), gmc.modules.BatchNorm(n_layers=l.n_feature_layers))
            elif config.bn_type == Config.BN_TYPE_COVARIANCE:
                norm = gmc.modules.CovScaleNorm(n_layers=l.n_feature_layers)
            elif config.bn_type == Config.BN_TYPE_STD:
                norm = gmc.modules.BatchNorm(n_layers=l.n_feature_layers)
            else:
                norm = nn.Module()
            self.norms.append(norm)
            n_feature_channels_in = l.n_feature_layers

        # self.weight_norm0 = prototype_modules.CentroidWeightNorm(batch_norm=False)
        # self.weight_norm = prototype_modules.CentroidWeightNorm(batch_norm=True)

        self.normX = nn.BatchNorm1d(num_features=n_feature_channels_in)

        if config.mlp is not None:
            mlp = list()
            for l in config.mlp:
                if l == -1:
                    mlp.append(nn.Dropout(p=0.5))
                else:
                    mlp.append(nn.Linear(n_feature_channels_in, l))
                    n_feature_channels_in = l
                    mlp.append(nn.ReLU())
                    mlp.append(nn.BatchNorm1d(n_feature_channels_in))
            self.mlp = nn.Sequential(*mlp)
        else:
            self.mlp = None

        self.timings = dict()
        self.last_time = time.time()

    def reset_timer(self):
        self.last_time = time.perf_counter()

    def time_lap(self, name: str):
        current = time.perf_counter()
        self.timings[name] = current - self.last_time
        self.last_time = current

    def set_position_learning(self, flag: bool):
        for gmc in self.gmcs:
            gmc.learn_positions = flag

    def set_covariance_learning(self, flag: bool):
        for gmc in self.gmcs:
            gmc.learn_covariances = flag

    def regularisation_loss(self) -> Tensor:
        rl = torch.zeros(1, device=next(self.parameters()).device, dtype=torch.float)
        for gmc in self.gmcs:
            rl = rl + gmc.regularisation_loss()
        return rl

    def weight_decay_loss(self) -> Tensor:
        wdl = torch.zeros(1, device=next(self.parameters()).device, dtype=torch.float)
        for gmc in self.gmcs:
            wdl = wdl + gmc.weight_decay_loss()
        return wdl

    # noinspection PyCallingNonCallable
    def forward(self, in_x: Tensor, tensorboard: TensorboardWriter = None) -> Tensor:
        # Andrew Ng says that most of the time batch norm (BN) is applied before activation.
        # That would allow to merge the beta and bias learnable parameters
        # https://www.youtube.com/watch?v=tNIpEZLv_eg
        # Other sources recommend to apply BN after the activation function.
        #
        # in our case: BN just scales and centres. the constant input to BN is ignored, so the constant convolution would be ignored if we place BN before ReLU.
        # but that might perform better anyway, we'll have to test.
        x, x_const = self.norm0((in_x, None))
        n_batch = gm.n_batch(x)

        for i in range(len(self.config.layers)):
            if gm.n_layers(x) > 8:
                x, x_const = self.droput((x, x_const))

            x, x_const = self.gmcs[i](x, x_const)

            if self.config.bn_place == Config.BN_PLACE_AFTER_GMC:
                x, x_const = self.norms[i]((x, x_const))

            if self.config.bias_type == Config.BIAS_TYPE_NEGATIVE_SOFTPLUS:
                x_const = x_const - F.softplus(self.biases[i], beta=20)
            elif self.config.bias_type == Config.BIAS_TYPE_NORMAL:
                x_const = x_const + self.biases[i]
            else:
                assert self.config.bias_type == Config.BIAS_TYPE_NONE
                x_const = torch.zeros(1, 1, device=in_x.device)

            self.reset_timer()
            x, x_const = self.relus[i](x, x_const, tensorboard)
            self.time_lap(f"relu{i}")
            # x = self.maxPool1(x)

            if self.config.bn_place == Config.BN_PLACE_AFTER_RELU:
                x, x_const = self.norms[i]((x, x_const))

        x = gm.integrate(x)
        x = self.normX(x)

        if self.mlp is not None:
            x = self.mlp(x)
        x = F.log_softmax(x, dim=1)
        return x.view(n_batch, self.config.n_classes)
