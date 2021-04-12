from __future__ import print_function
import pathlib
import time
import typing
import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter as TensorboardWriter

from qm9.config import Config
import gmc.mixture as gm
import gmc.modules as gmc_modules
import gmc.mat_tools as mat_tools
import qm9.prototype_modules as prototype_modules


class BatchNormStack:
    def __init__(self, norm_list: typing.Tuple):
        self.norm_list = norm_list

    def __call__(self, x: torch.Tensor, x_const: torch.Tensor = None):
        y = x
        y_const = x_const
        for norm in self.norm_list:
            y, y_const = norm(y, y_const)

        return y, y_const


class Net(nn.Module):
    def __init__(self,
                 name: str = "default",
                 learn_positions: bool = False,
                 learn_covariances: bool = False,
                 config: Config = Config):
        super(Net, self).__init__()
        self.storage_path = config.data_base_path / "weights" / f"mnist3d_gmcnet_{name}.pt"
        # reference_fitter = gmc_modules.generate_default_fitting_module

        self.config = config

        bias_0 = 0.0
        if self.config.bias_type == Config.BIAS_TYPE_NEGATIVE_SOFTPLUS:
            bias_0 = -0.1

        radius2cov = lambda p: (p / 3) ** 2
        # will return a weight that would integrate to 1 with the given radius
        radius2weight = lambda r: 1 / (config.n_kernel_components * math.sqrt((math.pi * radius2cov(r) * 2) ** 3))  # integral(gaussian) = sqrt(..)

        self.biases = torch.nn.ParameterList()
        self.gmcs = torch.nn.modules.ModuleList()
        self.relus = torch.nn.modules.ModuleList()

        if config.layers[-1].n_feature_layers == -1:
            config.layers[-1].n_feature_layers = config.n_classes
        n_feature_layers_in = 5
        last_n_fitting_components = -1
        for i, l in enumerate(config.layers):
            self.gmcs.append(gmc_modules.Convolution(config.convolution_config, n_layers_in=n_feature_layers_in, n_layers_out=l.n_feature_layers, n_kernel_components=config.n_kernel_components,
                                                     position_range=l.kernel_radius, covariance_range=radius2cov(l.kernel_radius),
                                                     learn_positions=learn_positions, learn_covariances=learn_covariances,
                                                     weight_sd=0.5 * radius2weight(l.kernel_radius), weight_mean=0.05 * radius2weight(l.kernel_radius), n_dims=3))
            self.biases.append(torch.nn.Parameter(torch.zeros(1, l.n_feature_layers) + bias_0))
            self.relus.append(gmc_modules.ReLUFitting(config.relu_config, layer_id=f"{i}c", n_layers=l.n_feature_layers, n_output_gaussians=l.n_fitting_components))
            last_n_fitting_components = l.n_fitting_components
            n_feature_layers_in = l.n_feature_layers

        # self.norm0 = BatchNormStack((  # gmc_modules.CovScaleNorm(norm_over_batch=False),
        #                              prototype_modules.IntegralNorm(config, True),))

        if config.bn_type == Config.BN_TYPE_COVARIANCE_INTEGRAL:
            self.norm = BatchNormStack((gmc_modules.CovScaleNorm(),
                                        prototype_modules.IntegralNorm(config),))
        elif config.bn_type == Config.BN_TYPE_ONLY_COVARIANCE:
            self.norm = BatchNormStack((gmc_modules.CovScaleNorm(), ))
        elif config.bn_type == Config.BN_TYPE_ONLY_INTEGRAL:
            self.norm = BatchNormStack((prototype_modules.IntegralNorm(config),))
        elif config.bn_type == Config.BN_TYPE_INTEGRAL_COVARIANCE:
            self.norm = BatchNormStack((prototype_modules.IntegralNorm(config),
                                        gmc_modules.CovScaleNorm(),))
        # self.weight_norm0 = prototype_modules.CentroidWeightNorm(norm_over_batch=False)
        # self.weight_norm = prototype_modules.CentroidWeightNorm(norm_over_batch=True)

        if config.mlp is not None:
            if config.mlp[-1] == -1:
                config.mlp[-1] = config.n_classes

            if last_n_fitting_components == 1:
                n_feature_layers_in = n_feature_layers_in * 13

            mlp = list()
            for l in config.mlp:
                if l == -1:
                    mlp.append(nn.Dropout(p=config.mlp_dropout))
                else:
                    mlp.append(nn.BatchNorm1d(n_feature_layers_in))
                    mlp.append(nn.Linear(n_feature_layers_in, l))
                    n_feature_layers_in = l
                    mlp.append(nn.ReLU())
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
        # x, x_const = self.norm0(in_x)
        x = in_x
        x_const = torch.zeros(1, 1, device=x.device)

        for i in range(len(self.config.layers)):
            if self.config.dataDropout > 0:
                if self.training:
                    n_selected_components = int(gm.n_components(x) * (1.0 - self.config.dataDropout))
                    indices = list()
                    for l in range(gm.n_layers(x)):
                        indices.append(torch.randperm(gm.n_components(x))[:n_selected_components].view(1, 1, -1))
                    indices = torch.cat(indices, 1)
                    x = mat_tools.my_index_select(x, indices.to(device=x.device))
                else:
                    weights = gm.weights(x) * (1.0 - self.config.dataDropout)
                    x = gm.pack_mixture(weights, gm.positions(x), gm.covariances(x))

            x, x_const = self.gmcs[i](x, x_const)

            if self.config.bn_place == Config.BN_PLACE_AFTER_GMC:
                x, x_const = self.norm(x, x_const)

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
                x, x_const = self.norm(x, x_const)

        if gm.n_components(x) > 1 or self.mlp is None:
            x = gm.integrate(x)
        else:
            x = x.view(gm.n_batch(x), -1)

        if self.mlp is not None:
            x = self.mlp(x)
        return x.view(-1)

    def save_model(self):
        print(f"experiment_model.Net.save: saving model to {self.storage_path}")
        whole_state_dict = self.state_dict()
        filtered_state = dict()
        for name, param in whole_state_dict.items():
            if "gm_fitting_net_666" not in name:
                filtered_state[name] = param
        torch.save(self.state_dict(), self.storage_path)

    # will load kernels and biases and fitting net params (if available)
    def load(self):
        print(f"experiment_model.Net.load: trying to load {self.storage_path}")
        if pathlib.Path(self.storage_path).is_file():
            whole_state_dict = torch.load(self.storage_path, map_location=torch.device('cpu'))
            filtered_state = dict()
            for name, param in whole_state_dict.items():
                if "gm_fitting_net_666" not in name:
                    filtered_state[name] = param

            missing_keys, unexpected_keys = self.load_state_dict(filtered_state, strict=False)
            print(f"experiment_model.Net.load: loaded (missing: {missing_keys}")  # we routinely have unexpected keys due to filtering
        else:
            print("experiment_model.Net.load: not found")

    def to(self, *args, **kwargs):
        return super(Net, self).to(*args, **kwargs)
