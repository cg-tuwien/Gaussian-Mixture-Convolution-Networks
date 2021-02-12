import pathlib
import os
import typing

import gmc.modules

source_dir = os.path.dirname(__file__)
data_base_path = pathlib.Path(f"{source_dir}/../../data")
modelnet_data_path = pathlib.Path(f"{data_base_path}/modelnet/gmms/fpsmax64_2")
modelnet_category_list_file = pathlib.Path(f"{data_base_path}/modelnet/pointclouds/modelnet10_shape_names.txt")
modelnet_training_sample_names_file = pathlib.Path(f"{data_base_path}/modelnet/pointclouds/modelnet10_train.txt")
modelnet_test_sample_names_file = pathlib.Path(f"{data_base_path}/modelnet/pointclouds/modelnet10_test.txt")

num_dataloader_workers = 0   # 0 -> main thread, otherwise number of threads. no auto available.
batch_size = 100

# mnist_n_in_g = 25
# mnist_n_layers_1 = 10
# mnist_n_out_g_1 = 24
# mnist_n_layers_2 = 12
# mnist_n_out_g_2 = 24
# mnist_n_out_g_3 = 12


n_kernel_components = 5

bn_mean_over_layers = False
BN_CONSTANT_COMPUTATION_ZERO = 0
BN_CONSTANT_COMPUTATION_MEAN_IN_CONST = 1
BN_CONSTANT_COMPUTATION_INTEGRAL = 2
BN_CONSTANT_COMPUTATION_WEIGHTED = 3
bn_constant_computation = BN_CONSTANT_COMPUTATION_ZERO

BN_PLACE_NOWHERE = 0
BN_PLACE_AFTER_GMC = 1
BN_PLACE_AFTER_RELU = 2
bn_place = BN_PLACE_AFTER_RELU

BIAS_TYPE_NONE = 0
BIAS_TYPE_NORMAL = 1
BIAS_TYPE_NEGATIVE_SOFTPLUS = 2
bias_type = BIAS_TYPE_NORMAL


class Layer:
    def __init__(self, n_feature_maps, kernel_radius, n_fitting_components):
        self.n_feature_layers = n_feature_maps
        self.kernel_radius = kernel_radius
        self.n_fitting_components = n_fitting_components


layers: typing.List[Layer] = [Layer(8, 1, 32),
                              Layer(16, 1, 16),
                              Layer(-1, 1, -1)]


def produce_name(layers: typing.List[Layer]) -> str:
    name = "L"
    for l in layers:
        name = f"{name}_{l.n_feature_layers}f_{int(l.kernel_radius * 10)}r"
    return name


relu_config: gmc.modules.ReLUFittingConfig = gmc.modules.ReLUFittingConfig()
convolution_config: gmc.modules.ConvolutionConfig = gmc.modules.ConvolutionConfig()

fitting_test_data_store_at_epoch = 10000
fitting_test_data_store_n_batches = 10
fitting_test_data_store_path = f"{data_base_path}/modelnet/fitting_input"
