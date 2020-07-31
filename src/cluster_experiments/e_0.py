import sys
import prototype_convolution.experiment_gm_mnist as experiment_gm_mnist

device = list(sys.argv)[1]
#device = "cuda"

experiment_gm_mnist.experiment(device=device, n_epochs=200, desc_string="k001", kernel_learning_rate=0.001, learn_covariances_after=200,
                               learn_positions_after=200, log_interval=1000, use_bias=True, batch_norm_per_layer=True)
