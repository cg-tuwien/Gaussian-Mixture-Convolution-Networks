import sys

import torchvision.datasets

import gmc.fitting
from gmc.model import Layer, Config as ModelConfig
import mnist_classification.main as main
from mnist_classification.config import Config

# device = list(sys.argv)[1]
device = "cuda"

c: Config = Config()
c.model.relu_config.fitting_method = gmc.fitting.splitter_and_fixed_point
c.input_fitting_iterations = 100
c.input_fitting_components = 64
c.model.bn_type = ModelConfig.BN_TYPE_COVARIANCE
c.model.bn_place = ModelConfig.BN_PLACE_AFTER_RELU
c.model.convolution_config.dropout = 0.0
c.model.dataDropout = 0.0
# c.model.relu_config.fitting_method = gmc.fitting.fixed_point_and_tree_hem2

c.log_tensorboard_renderings = False
c.n_epochs = 20
c.batch_size = 100
c.log_interval = 10000

# network size
c.model.layers = [Layer(8, 2.5, 96, 128),
                  Layer(16, 2.5, 48, 64),
                  Layer(32, 2.5, 24, 32),
                  Layer(64, 2.5, 12, 16),
                  # Layer(128, 2.5, 4),
                  # Layer(256, 2.5, 2),
                  # Layer(512, 2.5, 4),
                  Layer(10, 2.5, 2, -1)]
# c.model.mlp = (-1, 10)

# c.fitting_test_data_store_at_epoch = 0
# c.fitting_test_data_store_n_batches = 5
#
# c.dataset_class = torchvision.datasets.FashionMNIST
# c.dataset_name = "fashion_mnist"

# c.training_set_start = 10000
# c.training_set_end = 29000
# c.test_set_start = 5000
# c.test_set_end = 5000

main.experiment(device=device, desc_string=f"{c.produce_description()}", config=c)

c.model.bn_type = ModelConfig.BN_TYPE_COVARIANCE_STD
main.experiment(device=device, desc_string=f"{c.produce_description()}", config=c)

c.model.layers = [Layer(8, 2.5, 128, 160),
                  Layer(16, 2.5, 64, 96),
                  Layer(32, 2.5, 32, 48),
                  Layer(64, 2.5, 16, 24),
                  # Layer(128, 2.5, 4),
                  # Layer(256, 2.5, 2),
                  # Layer(512, 2.5, 4),
                  Layer(10, 2.5, 2, -1)]
main.experiment(device=device, desc_string=f"{c.produce_description()}", config=c)
