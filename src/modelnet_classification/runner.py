import sys
import modelnet_classification.main as main
import modelnet_classification.config as Config

# device = list(sys.argv)[1]
device = "cuda"

c: Config = Config
# output / test
c.fitting_test_data_store_at_epoch = 2000

# network size
c.mnist_n_in_g = 32
c.mnist_n_layers_1 = 8
c.mnist_n_out_g_1 = 32
c.mnist_n_layers_2 = 16
c.mnist_n_out_g_2 = 16
c.mnist_n_out_g_3 = -1

c.kernel_radius = 0.5

mnist_n_kernel_components = 5

# other network settings
c.bias_type = c.BIAS_TYPE_NONE
c.bn_place = c.BN_PLACE_BEFORE_GMC

# performance
c.batch_size = 21
c.num_dataloader_workers = 0

main.experiment(device=device, n_epochs=200, desc_string=f"minibatch_and_weightDecay_{c.mnist_n_layers_1}_{c.mnist_n_layers_2}_10_r{int(c.kernel_radius * 10)}", kernel_learning_rate=0.001, learn_covariances_after=0, learn_positions_after=0, log_interval=c.batch_size * 10, config=c)
