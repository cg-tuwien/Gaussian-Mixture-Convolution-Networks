import pathlib
from multiprocessing import Process
import copy

import gmc.fitting
from gmc.model import Layer, Config as ModelConfig

import mnist_classification.main as main
from mnist_classification.config import Config

# device = list(sys.argv)[1]
device = "cuda"

c: Config = Config()
c.input_fitting_iterations = 100
c.input_fitting_components = 8
c.model.bn_type = ModelConfig.BN_TYPE_COVARIANCE_STD
c.model.bn_place = ModelConfig.BN_PLACE_AFTER_RELU
c.model.convolution_config.dropout = 0.0
c.model.dataDropout = 0.0
c.model.relu_config.fitting_method = gmc.fitting.fixed_point_and_tree_hem2

# c.log_tensorboard_renderings = False
c.n_epochs = 32
c.batch_size = 40
c.log_interval = 1000
c.weight_decay_rate = 0.1

# network size
c.model.layers = [Layer(8, 1.5, 4),
                  Layer(16, 2.0, 2),
                  Layer(32, 2.5, 2),
                  Layer(10, 2.5, -1)]
main.experiment(device=device, desc_string=f"{c.produce_description()}", config=c, ablation_name="mnist_n_gaussians")

c.model.layers = [Layer(8, 1.5, 8),
                  Layer(16, 2.0, 4),
                  Layer(32, 2.5, 2),
                  Layer(10, 2.5, -1)]
main.experiment(device=device, desc_string=f"{c.produce_description()}", config=c, ablation_name="mnist_n_gaussians")
