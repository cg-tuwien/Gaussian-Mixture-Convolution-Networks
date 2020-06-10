from experiment_dl_conf import *

experiment_gm_mnist.experiment_alternating(device=list(sys.argv)[1], n_epochs=200, n_epochs_fitting_training=2, desc_string="_pS_lrnAmp",
                                             kernel_learning_rate=learning_rate_kernels, fitting_learning_rate=learning_rate_fitting,
                                             layer1_m2m_fitting=generate_fitting_module_S,
                                             layer2_m2m_fitting=generate_fitting_module_S,
                                             layer3_m2m_fitting=generate_fitting_module_S,
                                             learn_covariances_after=200, learn_positions_after=200,
                                             log_interval=log_interval)
