import prototype_convolution.runner as cluster
import prototype_convolution.config

# device = list(sys.argv)[1]
device = "cuda"

gmcn_config: prototype_convolution.config = cluster.default_gmcn_config()

gmcn_config.fitting_test_data_store_at_epoch = 20000
gmcn_config.fitting_test_data_store_path = "/home/madam/Documents/work/tuw/gmc_net/data/fitting_input_large2"
gmcn_config.mnist_n_in_g = 25
gmcn_config.mnist_n_layers_1 = 16
gmcn_config.mnist_n_out_g_1 = 16
gmcn_config.mnist_n_layers_2 = 32
gmcn_config.mnist_n_out_g_2 = 8
gmcn_config.mnist_n_out_g_3 = -1
gmcn_config.mnist_n_kernel_components = 3

gmcn_config.bias_type = gmcn_config.BIAS_TYPE_NONE
gmcn_config.bn_place = gmcn_config.BN_PLACE_BEFORE_GMC


cluster.run_with(device, "L3", gmcn_config)
