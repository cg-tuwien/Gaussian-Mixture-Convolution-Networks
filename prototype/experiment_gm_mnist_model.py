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
                 layer1_m2m_fitting: typing.Callable = gm_modules.generate_default_fitting_module,
                 layer2_m2m_fitting: typing.Callable = gm_modules.generate_default_fitting_module,
                 layer3_m2m_fitting: typing.Callable = gm_modules.generate_default_fitting_module,
                 n_kernel_components: int = config.mnist_n_kernel_components):
        super(Net, self).__init__()
        self.storage_path = config.data_base_path / "weights" / "mnist_gmcnet.pt"
        reference_fitter = gm_modules.generate_default_fitting_module
        n_in_g = config.mnist_n_in_g
        n_layers_1 = config.mnist_n_layers_1
        n_out_g_1 = config.mnist_n_out_g_1
        n_layers_2 = config.mnist_n_layers_2
        n_out_g_2 = config.mnist_n_out_g_2
        n_out_g_3 = config.mnist_n_out_g_3

        self.gmc1 = gm_modules.GmConvolution(n_layers_in=1, n_layers_out=n_layers_1, n_kernel_components=n_kernel_components,
                                             position_range=2, covariance_range=0.5,
                                             learn_positions=False, learn_covariances=False,
                                             weight_sd=0.4).cuda()
        self.relu1_reference = gm_modules.GmBiasAndRelu(layer_id="1r", n_layers=n_layers_1, generate_fitting_module=reference_fitter, n_input_gaussians=n_in_g * n_kernel_components, n_output_gaussians=n_out_g_1).cuda()
        self.relu1_current = gm_modules.GmBiasAndRelu(layer_id="1c", n_layers=n_layers_1, generate_fitting_module=layer1_m2m_fitting, n_input_gaussians=n_in_g * n_kernel_components, n_output_gaussians=n_out_g_1).cuda()
        # self.maxPool1 = gm_modules.MaxPooling(10)

        self.gmc2 = gm_modules.GmConvolution(n_layers_in=n_layers_1, n_layers_out=n_layers_2, n_kernel_components=n_kernel_components,
                                             position_range=4, covariance_range=2,
                                             learn_positions=False, learn_covariances=False,
                                             weight_sd=0.04).cuda()
        self.relu2_reference = gm_modules.GmBiasAndRelu(layer_id="2r", n_layers=n_layers_2, generate_fitting_module=reference_fitter, n_input_gaussians=n_out_g_1 * n_layers_1 * n_kernel_components,
                                                        n_output_gaussians=n_out_g_2).cuda()
        self.relu2_current = gm_modules.GmBiasAndRelu(layer_id="2c", n_layers=n_layers_2, generate_fitting_module=layer2_m2m_fitting, n_input_gaussians=n_out_g_1 * n_layers_1 * n_kernel_components,
                                                      n_output_gaussians=n_out_g_2).cuda()
        # self.maxPool2 = gm_modules.MaxPooling(10)

        self.gmc3 = gm_modules.GmConvolution(n_layers_in=n_layers_2, n_layers_out=10, n_kernel_components=n_kernel_components,
                                             position_range=8, covariance_range=4,
                                             learn_positions=False, learn_covariances=False,
                                             weight_sd=0.025).cuda()
        # self.relu3_reference = gm_modules.GmBiasAndRelu(layer_id="3r", n_layers=10, generate_fitting_module=reference_fitter, n_input_gaussians=n_out_g_2*n_layers_2*n_kernel_components, n_output_gaussians=n_out_g_3).cuda()
        self.relu3_current = gm_modules.GmBiasAndRelu(layer_id="3c", n_layers=10, generate_fitting_module=layer3_m2m_fitting, n_input_gaussians=n_out_g_2 * n_layers_2 * n_kernel_components,
                                                      n_output_gaussians=n_out_g_3).cuda()
        # self.maxPool3 = gm_modules.MaxPooling(2)

        self.bn0 = gm_modules.BatchNorm(per_gaussian_norm=True)
        self.bn = gm_modules.BatchNorm(per_gaussian_norm=False)

        self.relu1_sampler = gm_fitting.Sampler(self.relu1_current, n_training_samples=400)
        self.relu2_sampler = gm_fitting.Sampler(self.relu2_current, n_training_samples=400)
        self.relu3_sampler = gm_fitting.Sampler(self.relu3_current, n_training_samples=400)

    def set_fitting_training(self, flag: bool):
        self.relu1_current.train_fitting(flag)
        self.relu2_current.train_fitting(flag)
        self.relu3_current.train_fitting(flag)

    def run_fitting_sampling(self, in_x: torch.Tensor, sampling_layers, train: bool, epoch: int, tensor_board_writer: torch.utils.tensorboard.SummaryWriter):
        assert sampling_layers is not None and self.training
        in_x_norm = self.bn0(in_x)
        x = self.gmc1(in_x_norm.detach())
        if 1 in sampling_layers:
            # dirty hack: also train with random bias
            training_bias = self.relu1_current.bias.detach().clone()
            random_selection = torch.rand_like(training_bias, dtype=torch.float32) > 0.5  # no hay rand_like con dtype=torch.bool
            training_bias[random_selection] = torch.rand(int(random_selection.sum().item()), dtype=torch.float32, device=training_bias.device) * gm.weights(x.detach()).max()
            self.relu1_sampler.run_on(x.detach(), training_bias, epoch, train=train, tensor_board_writer=tensor_board_writer)
            if epoch % 40 == 0 and train:
                self.relu1_current.save_fitting_parameters()
                self.relu1_sampler.save_optimiser_state()

        x = self.relu1_reference(x)
        x = self.bn(x)
        # x = self.maxPool1(x)
        x = self.gmc2(x)

        if 2 in sampling_layers:
            training_bias = self.relu2_current.bias.detach().clone()
            random_selection = torch.rand_like(training_bias, dtype=torch.float32) > 0.5  # no hay rand_like con dtype=torch.bool
            training_bias[random_selection] = torch.rand(int(random_selection.sum().item()), dtype=torch.float32, device=training_bias.device) * gm.weights(x.detach()).max()
            self.relu2_sampler.run_on(x.detach(), training_bias, epoch, train=train, tensor_board_writer=tensor_board_writer)
            if epoch % 40 == 0 and train:
                self.relu2_current.save_fitting_parameters()
                self.relu2_sampler.save_optimiser_state()

        x = self.relu2_reference(x)
        x = self.bn(x)
        # x = self.maxPool2(x)
        x = self.gmc3(x)

        if 3 in sampling_layers:
            training_bias = self.relu3_current.bias.detach().clone()
            random_selection = torch.rand_like(training_bias, dtype=torch.float32) > 0.5  # no hay rand_like con dtype=torch.bool
            training_bias[random_selection] = torch.rand(int(random_selection.sum().item()), dtype=torch.float32, device=training_bias.device) * gm.weights(x.detach()).max()
            self.relu3_sampler.run_on(x.detach(), training_bias, epoch, train=train, tensor_board_writer=tensor_board_writer)
            if epoch % 40 == 0 and train:
                self.relu3_current.save_fitting_parameters()
                self.relu3_sampler.save_optimiser_state()

    def forward(self, in_x: torch.Tensor):
        in_x_norm = self.bn0(in_x)

        x = self.gmc1(in_x_norm)
        x = self.relu1_current(x)
        x = self.bn(x)
        # x = self.maxPool1(x)

        x = self.gmc2(x)
        x = self.relu2_current(x)
        x = self.bn(x)
        # x = self.maxPool2(x)

        x = self.gmc3(x)
        x = self.relu3_current(x)
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
        self.relu1_current.save_fitting_parameters()
        self.relu2_current.save_fitting_parameters()
        self.relu3_current.save_fitting_parameters()

    # will load kernels and biases and fitting net params (if available)
    def load(self):
        print(f"experiment_gm_mnist_model.Net.load: trying to load {self.storage_path}")
        if pathlib.Path(self.storage_path).is_file():
            whole_state_dict = torch.load(self.storage_path)
            filtered_state = dict()
            for name, param in whole_state_dict.items():
                if "gm_fitting_net_666" not in name:
                    filtered_state[name] = param

            missing_keys, unexpected_keys = self.load_state_dict(filtered_state, strict=False)
            print(f"experiment_gm_mnist_model.Net.load: loaded (missing: {missing_keys}, unexpected: {unexpected_keys}")
        else:
            print("experiment_gm_mnist_model.Net.load: not found")

        # warning, fitting must be loaded after the the state dict! this will overwrite the fitting params. so different
        # fitting params can be tested with the same kernels, biases and other params (if any)
        print("experiment_gm_mnist_model.Net.load: trying to load fitting params now")
        self.relu1_reference.load_fitting_parameters()
        self.relu2_reference.load_fitting_parameters()

        self.relu1_current.load_fitting_parameters()
        self.relu2_current.load_fitting_parameters()
        self.relu3_current.load_fitting_parameters()

        self.relu1_reference.bias = self.relu1_current.bias
        self.relu2_reference.bias = self.relu2_current.bias
