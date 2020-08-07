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

experiment_gm_mnist.experiment(device=device, n_epochs=200, desc_string="M3_k001", kernel_learning_rate=0.001, learn_covariances_after=200,
                               learn_positions_after=200, log_interval=1000, use_bias=True, batch_norm_per_layer=True, use_adam=True, gmcn_config=gmcn_config)
