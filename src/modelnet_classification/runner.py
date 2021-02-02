import sys
import modelnet_classification.main as main
import modelnet_classification.config as Config

# device = list(sys.argv)[1]
device = "cuda"

gmcn_config: Config = Config

gmcn_config.fitting_test_data_store_at_epoch = 2000

gmcn_config.mnist_n_in_g = 32
gmcn_config.mnist_n_layers_1 = 8
gmcn_config.mnist_n_out_g_1 = 16
gmcn_config.mnist_n_layers_2 = 10
gmcn_config.mnist_n_out_g_2 = 8
gmcn_config.mnist_n_out_g_3 = -1

mnist_n_kernel_components = 7

gmcn_config.bias_type = gmcn_config.BIAS_TYPE_NONE
gmcn_config.bn_place = gmcn_config.BN_PLACE_BEFORE_GMC


gmcn_config.num_dataloader_workers = 0

main.experiment(device=device, n_epochs=200, desc_string=f"modelnet1", kernel_learning_rate=0.001, learn_covariances_after=0, learn_positions_after=0, log_interval=100, config=gmcn_config)
