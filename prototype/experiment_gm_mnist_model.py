from __future__ import print_function
import pathlib
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import typing

import config
import gm
import gm_fitting
import gm_modules


class Net(nn.Module):
    def __init__(self,
                 name: str = "default",
                 layer1_m2m_fitting: typing.Callable = gm_modules.generate_default_fitting_module,
                 layer2_m2m_fitting: typing.Callable = gm_modules.generate_default_fitting_module,
                 layer3_m2m_fitting: typing.Callable = gm_modules.generate_default_fitting_module,
                 learn_positions: bool = False,
                 learn_covariances: bool = False,
                 n_kernel_components: int = config.mnist_n_kernel_components):
        super(Net, self).__init__()
        self.storage_path = config.data_base_path / "weights" / f"mnist_gmcnet_{name}.pt"
        # reference_fitter = gm_modules.generate_default_fitting_module
        n_in_g = config.mnist_n_in_g
        n_layers_1 = config.mnist_n_layers_1
        n_out_g_1 = config.mnist_n_out_g_1
        n_layers_2 = config.mnist_n_layers_2
        n_out_g_2 = config.mnist_n_out_g_2
        n_out_g_3 = config.mnist_n_out_g_3

        self.gmc1 = gm_modules.GmConvolution(n_layers_in=1, n_layers_out=n_layers_1, n_kernel_components=n_kernel_components,
                                             position_range=2, covariance_range=0.5,
                                             learn_positions=learn_positions, learn_covariances=learn_covariances,
                                             weight_sd=0.4)
        # self.maxPool1 = gm_modules.MaxPooling(10)

        self.gmc2 = gm_modules.GmConvolution(n_layers_in=n_layers_1, n_layers_out=n_layers_2, n_kernel_components=n_kernel_components,
                                             position_range=4, covariance_range=2,
                                             learn_positions=learn_positions, learn_covariances=learn_covariances,
                                             weight_sd=0.04)
        # self.maxPool2 = gm_modules.MaxPooling(10)

        self.gmc3 = gm_modules.GmConvolution(n_layers_in=n_layers_2, n_layers_out=10, n_kernel_components=n_kernel_components,
                                             position_range=8, covariance_range=4,
                                             learn_positions=learn_positions, learn_covariances=learn_covariances,
                                             weight_sd=0.025)
        # self.maxPool3 = gm_modules.MaxPooling(2)

        self.bn0 = gm_modules.BatchNorm(per_gaussian_norm=True)
        self.bn = gm_modules.BatchNorm(per_gaussian_norm=False)

        # initialise these last, so all the kernels should have the same random seed
        self.relus = torch.nn.modules.ModuleList()
        self.relus.append(gm_modules.GmBiasAndRelu(layer_id="1c", n_layers=n_layers_1, generate_fitting_module=layer1_m2m_fitting, n_input_gaussians=n_in_g * n_kernel_components, n_output_gaussians=n_out_g_1))
        self.relus.append(gm_modules.GmBiasAndRelu(layer_id="2c", n_layers=n_layers_2, generate_fitting_module=layer2_m2m_fitting, n_input_gaussians=n_out_g_1 * n_layers_1 * n_kernel_components, n_output_gaussians=n_out_g_2))
        self.relus.append(gm_modules.GmBiasAndRelu(layer_id="3c", n_layers=10, generate_fitting_module=layer3_m2m_fitting, n_input_gaussians=n_out_g_2 * n_layers_2 * n_kernel_components, n_output_gaussians=n_out_g_3))
        self.relus[0].fitting_sampler.load()
        self.relus[1].fitting_sampler.load()
        self.relus[2].fitting_sampler.load()

    def set_fitting_training(self, flag: bool):
        for relu in self.relus:
            relu.train_fitting(flag)
        self.gmc1.set_requires_grad(not flag)
        self.gmc2.set_requires_grad(not flag)
        self.gmc3.set_requires_grad(not flag)

    def set_position_learning(self, flag: bool):
        self.gmc1.learn_positions = flag
        self.gmc2.learn_positions = flag
        self.gmc3.learn_positions = flag

    def set_covariance_learning(self, flag: bool):
        self.gmc1.learn_covariances = flag
        self.gmc2.learn_covariances = flag
        self.gmc3.learn_covariances = flag

    def fitting_parameters(self):
        return list(self.relus[0].gm_fitting_net_666.parameters()) + \
               list(self.relus[1].gm_fitting_net_666.parameters()) + \
               list(self.relus[2].gm_fitting_net_666.parameters())

    def run_fitting_sampling(self, fitting_inputs: typing.List[torch.Tensor], train: bool, epoch: int,
                             tensor_board_writer: torch.utils.tensorboard.SummaryWriter, tensor_board_prefix: str = "") -> typing.List[torch.Tensor]:
        device=fitting_inputs[0].device
        losses = list()
        for i, relu in enumerate(self.relus):
            fitting_input = fitting_inputs[i]
            # training_bias_shape = list(relu.bias.shape)
            # training_bias_shape[0] = gm.n_batch(fitting_input)
            # std_dev = gm.weights(fitting_input).abs().mean().item()
            # training_bias = torch.normal(mean=0, std=std_dev, size=training_bias_shape, device=device).abs()
            training_bias = relu.bias.abs()
            losses.append(relu.fitting_sampler.run_on(fitting_input, training_bias, epoch, train=train, tensor_board_writer=tensor_board_writer, tensor_board_prefix=tensor_board_prefix))

        return losses

    def regularisation_loss(self):
        return self.gmc1.regularisation_loss() + self.gmc2.regularisation_loss() + self.gmc3.regularisation_loss()

    def forward(self, in_x: torch.Tensor, fitting_inputs: typing.List[torch.Tensor] = None):
        x = self.bn0(in_x)

        x = self.gmc1(x)
        if fitting_inputs is not None:
            fitting_inputs.append(x.detach())
        x = self.relus[0](x)
        x = self.bn(x)
        # x = self.maxPool1(x)

        x = self.gmc2(x)
        if fitting_inputs is not None:
            fitting_inputs.append(x.detach())
        x = self.relus[1](x)
        x = self.bn(x)
        # x = self.maxPool2(x)

        x = self.gmc3(x)
        if fitting_inputs is not None:
            fitting_inputs.append(x.detach())
        x = self.relus[2](x)
        x = self.bn(x)
        # x = self.maxPool3(x)

        x = gm.integrate(x)
        x = F.log_softmax(x, dim=1)
        return x.view(-1, 10)

    def save_model(self):
        print(f"experiment_gm_mnist_model.Net.save: saving model to {self.storage_path}")
        whole_state_dict = self.state_dict()
        filtered_state = dict()
        for name, param in whole_state_dict.items():
            if "gm_fitting_net_666" not in name:
                filtered_state[name] = param
        torch.save(self.state_dict(), self.storage_path)

    def save_fitting_parameters(self):
        print(f"experiment_gm_mnist_model.Net.save: saving fitting parameters")
        for relu in self.relus:
            relu.save_fitting_parameters()

    def save_fitting_optimiser_state(self):
        print(f"experiment_gm_mnist_model.Net.save: saving fitting parameters")
        for relu in self.relus:
            relu.fitting_sampler.save_optimiser_state()

    # will load kernels and biases and fitting net params (if available)
    def load(self):
        print(f"experiment_gm_mnist_model.Net.load: trying to load {self.storage_path}")
        if pathlib.Path(self.storage_path).is_file():
            whole_state_dict = torch.load(self.storage_path, map_location=torch.device('cpu'))
            filtered_state = dict()
            for name, param in whole_state_dict.items():
                if "gm_fitting_net_666" not in name:
                    filtered_state[name] = param

            missing_keys, unexpected_keys = self.load_state_dict(filtered_state, strict=False)
            print(f"experiment_gm_mnist_model.Net.load: loaded (missing: {missing_keys}")  # we routinely have unexpected keys due to filtering
        else:
            print("experiment_gm_mnist_model.Net.load: not found")

        # warning, fitting must be loaded after the the state dict! this will overwrite the fitting params. so different
        # fitting params can be tested with the same kernels, biases and other params (if any)
        print("experiment_gm_mnist_model.Net.load: trying to load fitting params now")
        for relu in self.relus:
            relu.load_fitting_parameters()

    def to(self, device: torch.device):
        for relu in self.relus:
            relu.fitting_sampler.to_(device)
        return super(Net, self).to(device)
