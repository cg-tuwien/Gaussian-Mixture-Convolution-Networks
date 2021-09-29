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
c.input_fitting_components = 128
c.model.bn_type = ModelConfig.BN_TYPE_COVARIANCE
c.model.bn_place = ModelConfig.BN_PLACE_AFTER_RELU
c.model.convolution_config.dropout = 0.0
c.model.dataDropout = 0.0
# c.model.relu_config.fitting_method = gmc.fitting.fixed_point_and_tree_hem2

c.log_tensorboard_renderings = True
c.n_epochs = 31
c.batch_size = 50
c.log_interval = 1000
c.dataset_class = torchvision.datasets.FashionMNIST
c.dataset_name = "fashion_mnist"

c.fitting_test_data_store_at_epoch = 100000

# for kernel_radius in (1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5):
#     c.model.layers = [Layer(8, kernel_radius, 192, 256),
#                       Layer(16, kernel_radius, 96, 128),
#                       Layer(32, kernel_radius, 48, 64),
#                       Layer(64, kernel_radius, 24, 32),
#                       Layer(128, kernel_radius, 12, 16),
#                       # Layer(256, kernel_radius, 6, -1),
#                       # Layer(512, 2.5, 4),
#                       Layer(10, kernel_radius, 2, -1)]
#     main.experiment(device=device, desc_string=f"{c.produce_description()}", config=c, ablation_name="radius")


c.input_fitting_components = 128
kernel_radius = 3.0
c.model.layers = [Layer(8, kernel_radius, 192, (224, 256)),
                  Layer(16, kernel_radius, 96, (112, 128)),
                  Layer(32, kernel_radius, 48, (56, 64)),
                  Layer(64, kernel_radius, 24, (28, 32)),
                  Layer(128, kernel_radius, 12, (14, 16)),
                  # Layer(256, kernel_radius, 6, -1),
                  # Layer(512, 2.5, 4),
                  Layer(10, kernel_radius, 2, -1)]
main.experiment(device=device, desc_string=f"{c.produce_description()}", config=c)

c.model.bn_place = ModelConfig.BN_PLACE_NOWHERE
c.model.layers = [Layer(8, kernel_radius, 192, 256),
                  Layer(16, kernel_radius, 96, 128),
                  Layer(32, kernel_radius, 48, 64),
                  Layer(64, kernel_radius, 24, 32),
                  Layer(128, kernel_radius, 12, 16),
                  # Layer(256, kernel_radius, 6, -1),
                  # Layer(512, 2.5, 4),
                  Layer(10, kernel_radius, 2, -1)]
c.model.bn_place = ModelConfig.BN_PLACE_NOWHERE
main.experiment(device=device, desc_string=f"{c.produce_description()}", config=c)

c.model.bn_place = ModelConfig.BN_PLACE_AFTER_GMC
main.experiment(device=device, desc_string=f"{c.produce_description()}", config=c)
