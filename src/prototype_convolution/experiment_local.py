# from experiment_dl_conf import *
#
# experiment_gm_mnist.experiment_alternating(device=list(sys.argv)[1], n_epochs=200, n_epochs_fitting_training=2, desc_string="_pS_lrnAmp",
#                                              kernel_learning_rate=learning_rate_kernels, fitting_learning_rate=learning_rate_fitting,
#                                              layer1_m2m_fitting=generate_fitting_module_S,
#                                              layer2_m2m_fitting=generate_fitting_module_S,
#                                              layer3_m2m_fitting=generate_fitting_module_S,
#                                              learn_covariances_after=200, learn_positions_after=200,
#                                              log_interval=log_interval)

import sys
import prototype_convolution.experiment_gm_mnist as experiment_gm_mnist
import prototype_convolution.config as gmcn_config

# device = list(sys.argv)[1]
device = "cuda"

gmcn_config.mnist_n_in_g = 25
gmcn_config.mnist_n_layers_1 = 8
gmcn_config.mnist_n_out_g_1 = 20
gmcn_config.mnist_n_layers_2 = 10
gmcn_config.mnist_n_out_g_2 = 8
gmcn_config.mnist_n_out_g_3 = 8

gmcn_config.bias_type = gmcn_config.BIAS_TYPE_NORMAL
gmcn_config.bn_constant_computation = gmcn_config.BN_CONSTANT_COMPUTATION_ZERO
gmcn_config.bn_mean_over_layers = False

gmcn_config.fitting_config.representative_select_mode = gmcn_config.fitting_config.REPRESENTATIVE_SELECT_MODE_RANDOM_TOP

experiment_gm_mnist.experiment(device=device, n_epochs=100, desc_string="M3_biasYes_bnCCzero_rndFitting_detachedInitM_fixed_klthr1.5", kernel_learning_rate=0.001, learn_covariances_after=100,
                               learn_positions_after=100, log_interval=1000, gmcn_config=gmcn_config)

# gmcn_config.bias_type = gmcn_config.BIAS_TYPE_NONE
# gmcn_config.bn_constant_computation = gmcn_config.BN_CONSTANT_COMPUTATION_INTEGRAL
# gmcn_config.bn_mean_over_layers = True
#
# experiment_gm_mnist.experiment(device=device, n_epochs=100, desc_string="M3_biasNone_bnCCintgrl_bnLayerMean", kernel_learning_rate=0.001, learn_covariances_after=200,
#                                learn_positions_after=200, log_interval=1000, gmcn_config=gmcn_config)
#
#
# gmcn_config.bias_type = gmcn_config.BIAS_TYPE_NORMAL
# gmcn_config.bn_constant_computation = gmcn_config.BN_CONSTANT_COMPUTATION_WEIGHTED
# gmcn_config.bn_mean_over_layers = False
#
# experiment_gm_mnist.experiment(device=device, n_epochs=100, desc_string="M3_biasYes_bnCCweightd", kernel_learning_rate=0.001, learn_covariances_after=100,
#                                learn_positions_after=100, log_interval=1000, gmcn_config=gmcn_config)
#
# gmcn_config.bias_type = gmcn_config.BIAS_TYPE_NORMAL
# gmcn_config.bn_constant_computation = gmcn_config.BN_CONSTANT_COMPUTATION_WEIGHTED
# gmcn_config.bn_mean_over_layers = True
#
# experiment_gm_mnist.experiment(device=device, n_epochs=100, desc_string="M3_biasYes_bnCCweightd_bnLayerMean", kernel_learning_rate=0.001, learn_covariances_after=100,
#                                learn_positions_after=100, log_interval=1000, gmcn_config=gmcn_config)


