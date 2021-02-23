from __future__ import print_function
import pathlib
import time
import typing

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter as TensorboardWriter

from modelnet_classification.config import Config
import gmc.mixture as gm
import gmc.modules as gmc_modules
import modelnet_classification.prototype_modules as prototype_modules


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

        pos2cov = lambda p: (p / 3) ** 2

        self.biases = torch.nn.ParameterList()
        self.gmcs = torch.nn.modules.ModuleList()
        self.relus = torch.nn.modules.ModuleList()

        config.layers[-1].n_feature_layers = config.n_classes
        n_feature_layers_in = 1
        for i, l in enumerate(config.layers):
            self.gmcs.append(gmc_modules.Convolution(config.convolution_config, n_layers_in=n_feature_layers_in, n_layers_out=l.n_feature_layers, n_kernel_components=config.n_kernel_components,
                                                     position_range=l.kernel_radius, covariance_range=pos2cov(l.kernel_radius),
                                                     learn_positions=learn_positions, learn_covariances=learn_covariances,
                                                     weight_sd=0.4, weight_mean=0.04, n_dims=3))
            self.biases.append(torch.nn.Parameter(torch.zeros(1, l.n_feature_layers) + bias_0))
            self.relus.append(gmc_modules.ReLUFitting(config.relu_config, layer_id=f"{i}c", n_layers=l.n_feature_layers, n_output_gaussians=l.n_fitting_components))
            n_feature_layers_in = l.n_feature_layers

        self.norm0 = BatchNormStack((gmc_modules.CovScaleNorm(norm_over_batch=False),
                                     prototype_modules.OldBatchNorm(config, True),))

        if config.bn_type == Config.BN_TYPE_COVARIANCE_INTEGRAL:
            self.norm = BatchNormStack((gmc_modules.CovScaleNorm(),
                                        prototype_modules.OldBatchNorm(config),))
        elif config.bn_type == Config.BN_TYPE_ONLY_COVARIANCE:
            self.norm = BatchNormStack((gmc_modules.CovScaleNorm(), ))
        elif config.bn_type == Config.BN_TYPE_ONLY_INTEGRAL:
            self.norm = BatchNormStack((prototype_modules.OldBatchNorm(config),))
        elif config.bn_type == Config.BN_TYPE_INTEGRAL_COVARIANCE:
            self.norm = BatchNormStack((prototype_modules.OldBatchNorm(config),
                                        gmc_modules.CovScaleNorm(),))
        # self.weight_norm0 = prototype_modules.CentroidWeightNorm(norm_over_batch=False)
        # self.weight_norm = prototype_modules.CentroidWeightNorm(norm_over_batch=True)

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
        x, x_const = self.norm0(in_x)

        for i in range(len(self.config.layers)):
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

        x = gm.integrate(x)
        x = F.log_softmax(x, dim=1)
        return x.view(-1, self.config.n_classes)

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
