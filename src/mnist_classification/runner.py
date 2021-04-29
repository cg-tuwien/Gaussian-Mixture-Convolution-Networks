import sys

import gmc.fitting
from gmc.model import Layer, Config as ModelConfig

import mnist_classification.main as main
from mnist_classification.config import Config

# device = list(sys.argv)[1]
device = "cuda"

c: Config = Config()
c.model.bn_type = ModelConfig.BN_TYPE_COVARIANCE_STD
c.model.bn_place = ModelConfig.BN_PLACE_AFTER_RELU
c.model.convolution_config.dropout = 0.0
c.model.dataDropout = 0.0
# c.model.relu_config.fitting_method = gmc.fitting.fixed_point_and_mhem

# c.log_tensorboard_renderings = False
c.n_epochs = 10
c.batch_size = 50
c.log_interval = 1000

# network size
c.model.layers = [Layer(8, 1.5, 28),
                  Layer(16, 2.0, 16),
                  Layer(32, 2.5, -1),]
c.model.mlp = (-1, 10)

main.experiment(device=device, desc_string=f"mnist_{c.produce_description()}", config=c)
