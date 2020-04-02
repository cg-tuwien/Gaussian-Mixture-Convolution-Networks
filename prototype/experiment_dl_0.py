from experiment_dl_conf import *

experiment_gm_mnist.experiment_alternating(device=list(sys.argv)[1], n_epochs=200, desc_string="_kd3pXS_lrnAll0",
                                             kernel_learning_rate=learning_rate_kernels, fitting_learning_rate=learning_rate_fitting,
                                             layer1_m2m_fitting=generate_subdivide_fitting_module(3, generate_fitting_module_XS),
                                             layer2_m2m_fitting=generate_subdivide_fitting_module(3, generate_fitting_module_XS),
                                             layer3_m2m_fitting=generate_subdivide_fitting_module(3, generate_fitting_module_XS),
                                             learn_covariances_after=0, learn_positions_after=0,
                                             log_interval=50)
