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
# c.data_base_path = pathlib.Path("/scratch/acelarek/gmms/")
c.input_fitting_iterations = 100
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
c.model.layers = [Layer(8, 1.5, 4),
                  Layer(16, 2.0, 2),
                  Layer(10, 2.5, -1)]
# c.model.mlp = (-1, 10)

c.training_set_start = 0
c.training_set_end = 0
for i in range(5):
    c.test_set_start = int(i * 2000)
    c.test_set_end = int((i+1) * 2000)
    Process(target=main.experiment, name=f"t{i}", kwargs={'device': device,
                                                          'desc_string': f"{c.produce_description()}",
                                                          "config": copy.deepcopy(c),
                                                          "ablation_name": "mnist_input_fitting"}).start()

# c.training_set_start = 0
# c.training_set_end = 0
# c.test_set_start = 0
# c.test_set_end = 10000
# Process(target=main.experiment, name="t10", kwargs={'device': device,
#                                                     'desc_string': f"{c.produce_description()}",
#                                                     "config": copy.deepcopy(c),
#                                                     "ablation_name": "mnist_input_fitting"}).start()
