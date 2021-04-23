import pathlib
import os
import sys
import typing

import gmc.modules


class Layer:
    def __init__(self, n_feature_maps, kernel_radius, n_fitting_components):
        self.n_feature_layers = n_feature_maps
        self.kernel_radius = kernel_radius
        self.n_fitting_components = n_fitting_components


def produce_name(layers: typing.List[Layer]) -> str:
    name = "L"
    for l in layers:
        name = f"{name}_{l.n_feature_layers}f_{int(l.kernel_radius * 10)}r_{int(l.n_fitting_components)}c"
    return name


class Config:
    BN_CONSTANT_COMPUTATION_ZERO = 0
    BN_CONSTANT_COMPUTATION_MEAN_IN_CONST = 1
    BN_CONSTANT_COMPUTATION_INTEGRAL = 2
    BN_CONSTANT_COMPUTATION_WEIGHTED = 3

    BIAS_TYPE_NONE = 0
    BIAS_TYPE_NORMAL = 1
    BIAS_TYPE_NEGATIVE_SOFTPLUS = 2

    BN_TYPE_ONLY_STD = "Std"
    BN_TYPE_ONLY_COVARIANCE = "Cov"
    BN_TYPE_COVARIANCE_STD = "CovStd"

    BN_PLACE_NOWHERE = "None"
    BN_PLACE_AFTER_GMC = "aCn"
    BN_PLACE_AFTER_RELU = "aRl"

    def __init__(self):
        # data sources
        self.source_dir = os.path.dirname(__file__)
        self.data_base_path = pathlib.Path(f"{self.source_dir}/../../data")

        # run settings
        self.num_dataloader_workers = 24   # 0 -> main thread, otherwise number of threads. no auto available.
        # https://stackoverflow.com/questions/38634988/check-if-program-runs-in-debug-mode
        if getattr(sys, 'gettrace', None) is not None and getattr(sys, 'gettrace')():
            # running in debugger
            self.num_dataloader_workers = 0

        self.n_classes = 10
        self.batch_size = 100
        self.n_epochs = 80
        self.kernel_learning_rate = 0.001
        self.weight_decay_rate = 0.05  # times kernel learning rate
        self.learn_covariances_after = 0
        self.learn_positions_after = 0

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
        self.dataDropout = 0.5

        self.relu_config: gmc.modules.ReLUFittingConfig = gmc.modules.ReLUFittingConfig()
        self.convolution_config: gmc.modules.ConvolutionConfig = gmc.modules.ConvolutionConfig(dropout=0.1)

        # logging
        self.save_model = False
        self.log_interval = 1000
        self.log_tensorboard_renderings = True
        self.fitting_test_data_store_at_epoch = 10000
        self.fitting_test_data_store_n_batches = 10
        self.fitting_test_data_store_path = f"{self.data_base_path}/modelnet/fitting_input"

    def produce_description(self):
        mlp_string = ""
        if self.mlp is not None:
            mlp_string = "_MLP"
            for l in self.mlp:
                mlp_string = f"{mlp_string}_{l}"
        return f"BN{self.bn_place}{self.bn_type}_cnDrp{int(self.convolution_config.dropout * 100)}_dtaDrp{int(self.dataDropout * 100)}_wDec{int(self.weight_decay_rate * 100)}_b{self.batch_size}_nK{self.n_kernel_components}_{produce_name(self.layers)}{mlp_string}_"

