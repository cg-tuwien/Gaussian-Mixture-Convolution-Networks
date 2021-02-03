import sys
import modelnet_classification.main as main
import modelnet_classification.config as Config

# device = list(sys.argv)[1]
device = "cuda"

gmcn_config: Config = Config
# output / test
gmcn_config.fitting_test_data_store_at_epoch = 2000

# network size
gmcn_config.mnist_n_in_g = 32
gmcn_config.mnist_n_layers_1 = 20
gmcn_config.mnist_n_out_g_1 = 32
gmcn_config.mnist_n_layers_2 = 16
gmcn_config.mnist_n_out_g_2 = 16
gmcn_config.mnist_n_out_g_3 = -1

mnist_n_kernel_components = 5

# other network settings
gmcn_config.bias_type = gmcn_config.BIAS_TYPE_NONE
gmcn_config.bn_place = gmcn_config.BN_PLACE_BEFORE_GMC

# performance
gmcn_config.batch_size = 95
gmcn_config.num_dataloader_workers = 0

main.experiment(device=device, n_epochs=200, desc_string=f"modelnet2", kernel_learning_rate=0.001, learn_covariances_after=0, learn_positions_after=0, log_interval=gmcn_config.batch_size*2, config=gmcn_config)
