import sys
import typing

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter as TensorboardWriter

import gmc.fitting
import gmc.mixture
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

c.model.relu_config.fitting_method = gmc.fitting.mse_testing_fitting_fun

# c.log_tensorboard_renderings = False
c.n_epochs = 160
c.batch_size = 50
c.log_interval = 1000

# network size
c.model.layers = [Layer(8, 1.5, 28),
                  Layer(16, 2.0, 16),
                  Layer(32, 2.5, 8),
                  Layer(10, 2.5, -1)]
# c.model.mlp = (-1, 10)

main.experiment(device=device, desc_string=f"{c.produce_description()}", config=c)
