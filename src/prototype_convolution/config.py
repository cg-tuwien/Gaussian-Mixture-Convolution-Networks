import pathlib
import os

import prototype_convolution.fitting

source_dir = os.path.dirname(__file__)
data_base_path = pathlib.Path(f"{source_dir}/../../data")
num_dataloader_workers = 0
batch_size = 100

# mnist_n_in_g = 25
# mnist_n_layers_1 = 10
# mnist_n_out_g_1 = 24
# mnist_n_layers_2 = 12
# mnist_n_out_g_2 = 24
# mnist_n_out_g_3 = 12

mnist_n_in_g = 25
mnist_n_layers_1 = 5
mnist_n_out_g_1 = 24
mnist_n_layers_2 = 6
mnist_n_out_g_2 = 12
mnist_n_out_g_3 = 6
mnist_n_kernel_components = 5

bn_mean_over_layers = False
BN_CONSTANT_COMPUTATION_ZERO = 0
BN_CONSTANT_COMPUTATION_MEAN_IN_CONST = 1
BN_CONSTANT_COMPUTATION_INTEGRAL = 2
BN_CONSTANT_COMPUTATION_WEIGHTED = 3
bn_constant_computation = BN_CONSTANT_COMPUTATION_ZERO

BN_PLACE_NOWHERE = 0
BN_PLACE_AFTER_GMC = 1
BN_PLACE_BEFORE_GMC = 2
bn_place = BN_PLACE_BEFORE_GMC

BIAS_TYPE_NONE = 0
BIAS_TYPE_NORMAL = 1
BIAS_TYPE_NEGATIVE_SOFTPLUS = 2
bias_type = BIAS_TYPE_NORMAL

fitting_method = prototype_convolution.fitting.fixed_point_and_mhem
fitting_config = prototype_convolution.fitting.Config()

fitting_test_data_store_at_epoch = 10000
fitting_test_data_store_n_batches = 10
fitting_test_data_store_path = "/home/madam/Documents/work/tuw/gmc_net/data/fitting_input"
