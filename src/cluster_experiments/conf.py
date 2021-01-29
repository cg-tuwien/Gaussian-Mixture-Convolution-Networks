import prototype_convolution.experiment_gm_mnist
import prototype_convolution.config


def default_gmcn_config() -> prototype_convolution.config:
    gmcn_config = prototype_convolution.config

    gmcn_config.mnist_n_in_g = 25
    gmcn_config.mnist_n_layers_1 = 8
    gmcn_config.mnist_n_out_g_1 = 32
    gmcn_config.mnist_n_layers_2 = 10
    gmcn_config.mnist_n_out_g_2 = 32
    gmcn_config.mnist_n_out_g_3 = -1

    gmcn_config.bn_mean_over_layers = False
    gmcn_config.bn_constant_computation = prototype_convolution.config.BN_CONSTANT_COMPUTATION_ZERO

    return gmcn_config


def run_with(device, name, gmcn_config):
    prototype_convolution.experiment_gm_mnist.experiment(device=device, n_epochs=11, desc_string=f"M3pp_{name}_bnCCzero_gAllFrom0", kernel_learning_rate=0.001, learn_covariances_after=0,
                                                         learn_positions_after=0, log_interval=5000, gmcn_config=gmcn_config)
