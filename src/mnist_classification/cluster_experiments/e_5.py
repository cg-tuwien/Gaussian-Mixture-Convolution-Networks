import sys

import gmc.fitting
from gmc.model import Layer, Config as ModelConfig

import mnist_classification.main as main
from mnist_classification.config import Config

# device = list(sys.argv)[1]
device = "cuda"

c: Config = Config()
c.input_fitting_iterations = 1
c.input_fitting_components = 64
c.model.bn_type = ModelConfig.BN_TYPE_COVARIANCE_STD
c.model.bn_place = ModelConfig.BN_PLACE_AFTER_RELU
c.model.convolution_config.dropout = 0.0
c.model.dataDropout = 0.0
c.model.relu_config.fitting_method = gmc.fitting.fixed_point_and_tree_hem2

# c.log_tensorboard_renderings = False
c.n_epochs = 10
c.batch_size = 50
c.log_interval = 1000

# network size
c.model.layers = [Layer(8, 1.5, 32),
                  Layer(16, 2.0, 16),
                  Layer(32, 2.5, 8),
                  Layer(64, 2.5, 4),
                  Layer(64, 2.5, 2),
                  Layer(10, 2.5, -1)]
# c.model.mlp = (-1, 10)

# c.training_set_start = 10000
# c.training_set_end = 11000
# c.test_set_start = 5000
# c.test_set_end = 5600

main.experiment(device=device, desc_string=f"{c.produce_description()}", config=c)
