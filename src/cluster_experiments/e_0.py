import sys
import prototype_convolution.experiment_gm_mnist as experiment_gm_mnist
import prototype_convolution.config as gmcn_config

device = list(sys.argv)[1]
#device = "cuda"

gmcn_config.mnist_n_in_g = 25
gmcn_config.mnist_n_layers_1 = 8
gmcn_config.mnist_n_out_g_1 = 20
gmcn_config.mnist_n_layers_2 = 10
gmcn_config.mnist_n_out_g_2 = 8
gmcn_config.mnist_n_out_g_3 = 8

gmcn_config.bias_type = gmcn_config.BIAS_TYPE_NONE
gmcn_config.bn_constant_computation = gmcn_config.BN_CONSTANT_COMPUTATION_ZERO
gmcn_config.bn_mean_over_layers = False

experiment_gm_mnist.experiment(device=device, n_epochs=100, desc_string="M3_biasNone_bnCCzero", kernel_learning_rate=0.001, learn_covariances_after=200,
                               learn_positions_after=200, log_interval=1000, gmcn_config=gmcn_config)

gmcn_config.bias_type = gmcn_config.BIAS_TYPE_NONE
gmcn_config.bn_constant_computation = gmcn_config.BN_CONSTANT_COMPUTATION_ZERO
gmcn_config.bn_mean_over_layers = True

experiment_gm_mnist.experiment(device=device, n_epochs=100, desc_string="M3_biasNone_bnCCzero_bnLayerMean", kernel_learning_rate=0.001, learn_covariances_after=200,
                               learn_positions_after=200, log_interval=1000, gmcn_config=gmcn_config)


gmcn_config.bias_type = gmcn_config.BIAS_TYPE_NONE
gmcn_config.bn_constant_computation = gmcn_config.BN_CONSTANT_COMPUTATION_ZERO
gmcn_config.bn_mean_over_layers = False

experiment_gm_mnist.experiment(device=device, n_epochs=100, desc_string="M3_biasNone_bnCCzero_learnAll50", kernel_learning_rate=0.001, learn_covariances_after=50,
                               learn_positions_after=50, log_interval=1000, gmcn_config=gmcn_config)

gmcn_config.bias_type = gmcn_config.BIAS_TYPE_NONE
gmcn_config.bn_constant_computation = gmcn_config.BN_CONSTANT_COMPUTATION_ZERO
gmcn_config.bn_mean_over_layers = True

experiment_gm_mnist.experiment(device=device, n_epochs=100, desc_string="M3_biasNone_bnCCzero_bnLayerMean_learnAll50", kernel_learning_rate=0.001, learn_covariances_after=50,
                               learn_positions_after=50, log_interval=1000, gmcn_config=gmcn_config)


