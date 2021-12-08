import sys

import torchvision.datasets

import gmc.fitting
from gmc.model import Layer, Config as ModelConfig
import mnist_classification.main as main
from mnist_classification.config import Config

# device = list(sys.argv)[1]
device = "cuda"

c: Config = Config()
c.model.relu_config.fitting_method = gmc.fitting.fixed_point_and_tree_hem
c.model.convolution_config.learnable_radius = False
c.input_fitting_iterations = 100
c.input_fitting_components = 128
c.model.bn_type = ModelConfig.BN_TYPE_COVARIANCE
c.model.bn_place = ModelConfig.BN_PLACE_AFTER_RELU
c.model.convolution_config.dropout = 0.0
c.model.dataDropout = 0.0
# c.model.relu_config.fitting_method = gmc.fitting.fixed_point_and_tree_hem2
c.dataset_class = torchvision.datasets.FashionMNIST
c.dataset_name = "fashion_mnist"

c.log_tensorboard_renderings = True
c.n_epochs = 61
c.batch_size = 50
c.log_interval = 1000

# network size
c.model.layers = [Layer(8, 1.50, 256, 64),
                  Layer(16, 1.50, 128, 32),
                  Layer(32, 1.5, 64, 16),
                  Layer(64, 1.5, 32, 8),
                  # Layer(128, 2.5, 16, 4),
                  # Layer(256, 2.5, 2),
                  # Layer(512, 2.5, 4),
                  # Layer(10, 2.2, 16, -1)
                  ]
c.model.mlp = (-1, 128, -1, 256, -1, 10)

# c.fitting_test_data_store_at_epoch = 0
# c.fitting_test_data_store_n_batches = 5
#
# c.dataset_class = torchvision.datasets.FashionMNIST
# c.dataset_name = "fashion_mnist"

# c.training_set_start = 10000
# c.training_set_end = 29000
# c.test_set_start = 5000
# c.test_set_end = 5000

main.experiment(device=device, desc_string=f"{c.produce_description()}", config=c, ablation_name=f"fashion_mnist")
#
# c.model.layers = [Layer(8, 1.5, 256, -1),
#                   Layer(16, 1.5, 128, -1),
#                   Layer(32, 1.5, 64, -1),
#                   Layer(64, 1.5, 32, -1),
#                   Layer(128, 1.5, 16, -1),
#                   # Layer(256, 2.5, 2),
#                   # Layer(512, 2.5, 4),
#                   Layer(10, 1.5, 16, -1)]
# # c.model.mlp = (-1, 10)
#
# # c.fitting_test_data_store_at_epoch = 0
# # c.fitting_test_data_store_n_batches = 5
# #
# c.dataset_class = torchvision.datasets.FashionMNIST
# c.dataset_name = "fashion_mnist"
#
# # c.training_set_start = 10000
# # c.training_set_end = 29000
# # c.test_set_start = 5000
# # c.test_set_end = 5000
#
# main.experiment(device=device, desc_string=f"{c.produce_description()}", config=c, ablation_name=f"fashion_mnist")
