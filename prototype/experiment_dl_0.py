from experiment_dl_conf import *

experiment_gm_mnist.experiment_probabalistic(device=list(sys.argv)[1], n_epochs=200, n_epochs_fitting_training=2, desc_string="_kd3pM_lrnCov35",
                                             kernel_learning_rate=learning_rate_kernels, fitting_learning_rate=learning_rate_fitting,
                                             layer1_m2m_fitting=generate_subdivide_fitting_module(3, generate_fitting_module_M),
                                             layer2_m2m_fitting=generate_subdivide_fitting_module(3, generate_fitting_module_M),
                                             layer3_m2m_fitting=generate_subdivide_fitting_module(3, generate_fitting_module_M),
                                             learn_covariances_after=35, learn_positions_after=200,
                                             log_interval=log_interval)
