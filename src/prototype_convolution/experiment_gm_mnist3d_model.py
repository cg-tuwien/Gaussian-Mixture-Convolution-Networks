from __future__ import print_function
import pathlib
import time

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter as TensorboardWriter

import prototype_convolution.config
import gmc.mixture as gm
import gmc.modules as gmc_modules
import prototype_convolution.modules as prototype_modules


class Net(nn.Module):
    def __init__(self,
                 name: str = "default",
                 learn_positions: bool = False,
                 learn_covariances: bool = False,
                 gmcn_config: prototype_convolution.config = prototype_convolution.config):
        super(Net, self).__init__()
        self.storage_path = gmcn_config.data_base_path / "weights" / f"mnist3d_gmcnet_{name}.pt"
        # reference_fitter = gmc_modules.generate_default_fitting_module
        n_in_g = gmcn_config.mnist_n_in_g
        n_layers_1 = gmcn_config.mnist_n_layers_1
        n_out_g_1 = gmcn_config.mnist_n_out_g_1
        n_layers_2 = gmcn_config.mnist_n_layers_2
        n_out_g_2 = gmcn_config.mnist_n_out_g_2
        n_out_g_3 = gmcn_config.mnist_n_out_g_3
        n_kernel_components = gmcn_config.mnist_n_kernel_components
        self.config = gmcn_config

        bias_0 = 0.0
        if self.config.bias_type == prototype_convolution.config.BIAS_TYPE_NEGATIVE_SOFTPLUS:
            bias_0 = -0.1

        self.biases = torch.nn.ParameterList()
        self.gmc1 = gmc_modules.Convolution(gmcn_config.convolution_config, n_layers_in=1, n_layers_out=n_layers_1, n_kernel_components=n_kernel_components,
                                            position_range=2, covariance_range=0.5,
                                            learn_positions=learn_positions, learn_covariances=learn_covariances,
                                            weight_sd=0.4, n_dims=3)
        self.biases.append(torch.nn.Parameter(torch.zeros(1, n_layers_1) + bias_0))
        # self.maxPool1 = gmc_modules.MaxPooling(10)

        self.gmc2 = gmc_modules.Convolution(gmcn_config.convolution_config, n_layers_in=n_layers_1, n_layers_out=n_layers_2, n_kernel_components=n_kernel_components,
                                            position_range=4, covariance_range=2,
                                            learn_positions=learn_positions, learn_covariances=learn_covariances,
                                            weight_sd=0.04, n_dims=3)
        self.biases.append(torch.nn.Parameter(torch.zeros(1, n_layers_2) + bias_0))
        # self.maxPool2 = gmc_modules.MaxPooling(10)

        self.gmc3 = gmc_modules.Convolution(gmcn_config.convolution_config, n_layers_in=n_layers_2, n_layers_out=10, n_kernel_components=n_kernel_components,
                                            position_range=8, covariance_range=4,
                                            learn_positions=learn_positions, learn_covariances=learn_covariances,
                                            weight_sd=0.025, n_dims=3)
        self.biases.append(torch.nn.Parameter(torch.zeros(1, 10) + bias_0))
        # self.maxPool3 = gmc_modules.MaxPooling(2)

        self.bn0 = prototype_modules.BatchNorm(gmcn_config, per_mixture_norm=True)
        self.bn = prototype_modules.BatchNorm(gmcn_config, per_mixture_norm=False)

        self.relus = torch.nn.modules.ModuleList()
        self.relus.append(gmc_modules.ReLUFitting(gmcn_config.relu_config, layer_id="1c", n_layers=n_layers_1, n_input_gaussians=n_in_g * n_kernel_components, n_output_gaussians=n_out_g_1))
        self.relus.append(gmc_modules.ReLUFitting(gmcn_config.relu_config, layer_id="2c", n_layers=n_layers_2, n_input_gaussians=n_out_g_1 * n_layers_1 * n_kernel_components, n_output_gaussians=n_out_g_2))
        self.relus.append(gmc_modules.ReLUFitting(gmcn_config.relu_config, layer_id="3c", n_layers=10, n_input_gaussians=n_out_g_2 * n_layers_2 * n_kernel_components, n_output_gaussians=n_out_g_3))

        self.timings = dict()
        self.last_time = time.time()

    def reset_timer(self):
        self.last_time = time.perf_counter()

    def time_lap(self, name: str):
        current = time.perf_counter()
        self.timings[name] = current - self.last_time
        self.last_time = current

    def set_position_learning(self, flag: bool):
        self.gmc1.learn_positions = flag
        self.gmc2.learn_positions = flag
        self.gmc3.learn_positions = flag

    def set_covariance_learning(self, flag: bool):
        self.gmc1.learn_covariances = flag
        self.gmc2.learn_covariances = flag
        self.gmc3.learn_covariances = flag

    def regularisation_loss(self):
        return self.gmc1.regularisation_loss() + self.gmc2.regularisation_loss() + self.gmc3.regularisation_loss()

    # noinspection PyCallingNonCallable
    def forward(self, in_x: Tensor, tensorboard: TensorboardWriter = None) -> Tensor:
        # Andrew Ng says that most of the time batch norm (BN) is applied before activation.
        # That would allow to merge the beta and bias learnable parameters
        # https://www.youtube.com/watch?v=tNIpEZLv_eg
        # Other sources recommend to apply BN after the activation function.
        #
        # in our case: BN just scales and centres. the constant input to BN is ignored, so the constant convolution would be ignored if we place BN before ReLU.
        # but that might perform better anyway, we'll have to test.
        x, x_const = self.bn0(in_x)
        x, x_const = self.gmc1(x, x_const)
        if self.config.bn_place == prototype_convolution.config.BN_PLACE_AFTER_GMC:
            x, x_const = self.bn(x, x_const)

        if self.config.bias_type == prototype_convolution.config.BIAS_TYPE_NEGATIVE_SOFTPLUS:
            x_const = x_const - F.softplus(self.biases[0], beta=20)
        elif self.config.bias_type == prototype_convolution.config.BIAS_TYPE_NORMAL:
            x_const = x_const + self.biases[0]
        else:
            assert self.config.bias_type == prototype_convolution.config.BIAS_TYPE_NONE
            x_const = torch.zeros(1, 1, device=in_x.device)

        self.reset_timer()
        x, x_const = self.relus[0](x, x_const, tensorboard)
        self.time_lap("relu0")
        # x = self.maxPool1(x)

        if self.config.bn_place == prototype_convolution.config.BN_PLACE_BEFORE_GMC:
            x, x_const = self.bn(x, x_const)
        x, x_const = self.gmc2(x, x_const)
        if self.config.bn_place == prototype_convolution.config.BN_PLACE_AFTER_GMC:
            x, x_const = self.bn(x, x_const)

        if self.config.bias_type == prototype_convolution.config.BIAS_TYPE_NEGATIVE_SOFTPLUS:
            x_const = x_const - F.softplus(self.biases[1], beta=20)
        elif self.config.bias_type == prototype_convolution.config.BIAS_TYPE_NORMAL:
            x_const = x_const + self.biases[1]
        else:
            assert self.config.bias_type == prototype_convolution.config.BIAS_TYPE_NONE
            x_const = torch.zeros(1, 1, device=in_x.device)

        self.reset_timer()
        x, x_const = self.relus[1](x, x_const, tensorboard)
        self.time_lap("relu1")
        # x = self.maxPool2(x)

        if self.config.bn_place == prototype_convolution.config.BN_PLACE_BEFORE_GMC:
            x, x_const = self.bn(x, x_const)
        x, x_const = self.gmc3(x, x_const)
        if self.config.bn_place == prototype_convolution.config.BN_PLACE_AFTER_GMC:
            x, x_const = self.bn(x, x_const)

        if self.config.bias_type == prototype_convolution.config.BIAS_TYPE_NEGATIVE_SOFTPLUS:
            x_const = x_const - F.softplus(self.biases[2], beta=20)
        elif self.config.bias_type == prototype_convolution.config.BIAS_TYPE_NORMAL:
            x_const = x_const + self.biases[2]
        else:
            assert self.config.bias_type == prototype_convolution.config.BIAS_TYPE_NONE
            x_const = torch.zeros(1, 1, device=in_x.device)

        self.reset_timer()
        x, x_const = self.relus[2](x, x_const, tensorboard)
        self.time_lap("relu2")
        # x = self.maxPool3(x)

        x = gm.integrate(x)
        x = F.log_softmax(x, dim=1)
        return x.view(-1, 10)

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
