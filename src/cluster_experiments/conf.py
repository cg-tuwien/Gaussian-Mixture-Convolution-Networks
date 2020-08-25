import prototype_convolution.experiment_gm_mnist


def run_with(device, name, gmcn_config):
    gmcn_config.mnist_n_in_g = 25
    gmcn_config.mnist_n_layers_1 = 8
    gmcn_config.mnist_n_out_g_1 = 20
    gmcn_config.mnist_n_layers_2 = 10
    gmcn_config.mnist_n_out_g_2 = 10
    gmcn_config.mnist_n_out_g_3 = -1

    gmcn_config.bn_mean_over_layers = False
    gmcn_config.bn_constant_computation = gmcn_config.BN_CONSTANT_COMPUTATION_ZERO

    prototype_convolution.experiment_gm_mnist.experiment(device=device, n_epochs=50, desc_string=f"M3pp_{name}_bnCCzero_gPosiOnly", kernel_learning_rate=0.001, learn_covariances_after=200,
                                                         learn_positions_after=200, log_interval=5000, gmcn_config=gmcn_config)
