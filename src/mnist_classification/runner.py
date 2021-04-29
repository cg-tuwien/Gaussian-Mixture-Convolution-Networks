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


def mse_testing_fitting_fun(mixture: Tensor, constant: Tensor, n_components: int, config: Config = Config(), tensorboard = None):
    if tensorboard is None:
        return gmc.fitting.fixed_point_and_tree_hem(mixture, constant, n_components, config)

    # run tests, they'll log to tensorboard
    gmc.fitting.fixed_point_and_mhem(mixture, constant, 8, config, (tensorboard[0][0], tensorboard[-1]))
    gmc.fitting.fixed_point_and_mhem(mixture, constant, 16, config, (tensorboard[0][1], tensorboard[-1]))
    gmc.fitting.fixed_point_and_mhem(mixture, constant, 32, config, (tensorboard[0][2], tensorboard[-1]))
    gmc.fitting.fixed_point_and_mhem(mixture, constant, 64, config, (tensorboard[0][3], tensorboard[-1]))
    if gmc.mixture.n_components(mixture) < 1280:     # out of memory
        gmc.fitting.fixed_point_and_mhem(mixture, constant, 128, config, (tensorboard[0][4], tensorboard[-1]))

    gmc.fitting.fixed_point_and_tree_hem(mixture, constant, 8, config, (tensorboard[0][5], tensorboard[-1]))
    gmc.fitting.fixed_point_and_tree_hem(mixture, constant, 16, config, (tensorboard[0][6], tensorboard[-1]))
    gmc.fitting.fixed_point_and_tree_hem(mixture, constant, 32, config, (tensorboard[0][7], tensorboard[-1]))
    gmc.fitting.fixed_point_and_tree_hem(mixture, constant, 64, config, (tensorboard[0][8], tensorboard[-1]))
    gmc.fitting.fixed_point_and_tree_hem(mixture, constant, 128, config, (tensorboard[0][9], tensorboard[-1]))

    # compute the mixture for the next level (ensuring the same input is used for measuring error of mhem and tree hem.
    return gmc.fitting.fixed_point_and_tree_hem(mixture, constant, n_components, config)


c.model.relu_config.fitting_method = mse_testing_fitting_fun

# c.log_tensorboard_renderings = False
c.n_epochs = 5
c.batch_size = 50
c.log_interval = 1000

# network size
c.model.layers = [Layer(8, 1.5, 28),
                  Layer(16, 2.0, 16),
                  Layer(32, 2.5, -1),]
c.model.mlp = (-1, 10)

main.experiment(device=device, desc_string=f"mnist_{c.produce_description()}", config=c)
