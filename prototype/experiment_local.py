from experiment_dl_conf import *

experiment_gm_mnist.experiment_probabalistic(device=list(sys.argv)[1], n_epochs=200, n_epochs_fitting_training=10, desc_string="_kd6pS_lrnAll25",
                                             kernel_learning_rate=learning_rate_kernels, fitting_learning_rate=learning_rate_fitting,
                                             layer1_m2m_fitting=generate_subdivide_fitting_module(6, generate_fitting_module_S),
                                             layer2_m2m_fitting=generate_subdivide_fitting_module(6, generate_fitting_module_S),
                                             layer3_m2m_fitting=generate_subdivide_fitting_module(6, generate_fitting_module_S),
                                             learn_covariances_after=30, learn_positions_after=30,
                                             log_interval=7000)
