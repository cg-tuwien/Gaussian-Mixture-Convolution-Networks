import pathlib
import os

source_dir = os.path.dirname(__file__)
data_base_path = pathlib.Path(f"{source_dir}/../../data")
num_dataloader_workers = 0
batch_size = 100

# mnist_n_in_g = 25
# mnist_n_layers_1 = 10
# mnist_n_out_g_1 = 24
# mnist_n_layers_2 = 12
# mnist_n_out_g_2 = 24
# mnist_n_out_g_3 = 12

mnist_n_in_g = 25
mnist_n_layers_1 = 5
mnist_n_out_g_1 = 24
mnist_n_layers_2 = 6
mnist_n_out_g_2 = 12
mnist_n_out_g_3 = 6
mnist_n_kernel_components = 5
