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
c.input_fitting_components = 64
c.n_epochs = 32
c.batch_size = 40
c.log_interval = 1000

c.model.layers = [Layer(8, 1.5, 32),
                  Layer(16, 2.0, 16),
                  Layer(10, 2.5, -1)]
main.experiment(device=device, desc_string=f"{c.produce_description()}", config=c, ablation_name="mnist_network_length_correct_norm")

