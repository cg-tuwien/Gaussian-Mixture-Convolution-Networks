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

gmcn_config.bias_type = gmcn_config.BIAS_TYPE_NORMAL
gmcn_config.bn_constant_computation = gmcn_config.BN_CONSTANT_COMPUTATION_WEIGHTED
gmcn_config.bn_mean_over_layers = False

experiment_gm_mnist.experiment(device=device, n_epochs=200, desc_string="nobias_noconst_bnOverLayers", kernel_learning_rate=0.001, learn_covariances_after=200,
                               learn_positions_after=200, log_interval=100, gmcn_config=gmcn_config)
