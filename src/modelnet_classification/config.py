import pathlib
import os
import sys
import typing

from gmc.model import Config as ModelConfig


class Config:

    def __init__(self, gmms_fitting: str = "fpsmax64_2", gengmm_path: typing.Optional[str] = None, n_classes: int = 10):
        # data sources
        self.source_dir = os.path.dirname(__file__)
        self.data_base_path = pathlib.Path(f"{self.source_dir}/../../data")
        if gengmm_path is None:
            self.modelnet_data_path = pathlib.Path(f"{self.data_base_path}/modelnet/gmms/{gmms_fitting}")
        else:
            self.modelnet_data_path = pathlib.Path(f"{gengmm_path}/{gmms_fitting}")
        self.modelnet_category_list_file = pathlib.Path(f"{self.data_base_path}/modelnet/pointclouds/modelnet{n_classes}_shape_names.txt")
        self.modelnet_training_sample_names_file = pathlib.Path(f"{self.data_base_path}/modelnet/pointclouds/modelnet{n_classes}_train.txt")
        self.modelnet_test_sample_names_file = pathlib.Path(f"{self.data_base_path}/modelnet/pointclouds/modelnet{n_classes}_test.txt")

        # run settings
        self.num_dataloader_workers = 24   # 0 -> main thread, otherwise number of threads. no auto available.
        # https://stackoverflow.com/questions/38634988/check-if-program-runs-in-debug-mode
        if getattr(sys, 'gettrace', None) is not None and getattr(sys, 'gettrace')():
            # running in debugger
            self.num_dataloader_workers = 0

        self.model = ModelConfig(n_dims=3)
        self.model.n_classes = n_classes
        self.n_classes = n_classes

        self.n_input_gaussians = -1
        self.batch_size = 21
        self.n_epochs = 162
        self.kernel_learning_rate = 0.001
        self.weight_decay_rate = 0.05  # times kernel learning rate
        self.learn_covariances_after = 0
        self.learn_positions_after = 0

        # logging
        self.save_model = False
        self.log_interval = self.batch_size * 10
        self.log_tensorboard_renderings = True
        self.fitting_test_data_store_at_epoch = 10000
        self.fitting_test_data_store_n_batches = 10
        self.fitting_test_data_store_path = f"{self.data_base_path}/modelnet/fitting_input"

    def produce_description(self):
        return f"lr{int(self.kernel_learning_rate * 1000)}_wDec{int(self.weight_decay_rate * 100)}_{self.model.produce_description()}"

