import pathlib
import os
import sys

from gmc.model import Config as ModelConfig


class Config:
    def __init__(self):
        # data sources
        self.source_dir = os.path.dirname(__file__)
        self.data_base_path = pathlib.Path(f"{self.source_dir}/../../data")

        # run settings
        self.num_dataloader_workers = 24   # 0 -> main thread, otherwise number of threads. no auto available.
        # https://stackoverflow.com/questions/38634988/check-if-program-runs-in-debug-mode
        if getattr(sys, 'gettrace', None) is not None and getattr(sys, 'gettrace')():
            # running in debugger
            self.num_dataloader_workers = 0

        self.model = ModelConfig(n_dims=2)
        self.model.n_classes = 10

        self.training_set_start = 0
        self.training_set_end = 60000
        self.test_set_start = 0
        self.test_set_end = 10000
        self.input_fitting_components = 32
        self.input_fitting_iterations = 100
        self.batch_size = 100
        self.n_epochs = 80
        self.kernel_learning_rate = 0.001
        self.weight_decay_rate = 0.05  # times kernel learning rate
        self.learn_covariances_after = 0
        self.learn_positions_after = 0

        # logging
        self.save_model = False
        self.log_interval = 1000
        self.log_tensorboard_renderings = True
        self.fitting_test_data_store_at_epoch = 10000
        self.fitting_test_data_store_n_batches = 10
        self.fitting_test_data_store_path = f"{self.data_base_path}/mnist/fitting_input"

    def produce_description(self):
        return f"{self.produce_input_description()}_lr{int(self.kernel_learning_rate * 1000)}_wDec{int(self.weight_decay_rate * 100)}_{self.model.produce_description()}"

    def produce_input_description(self):
        if self.input_fitting_iterations == 1:
            return f"mnist_init{self.input_fitting_components}"
        if self.input_fitting_iterations == -1:
            return "mnist"
        return f"mnist_em{self.input_fitting_components}"
