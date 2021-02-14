import sys
import modelnet_classification.main as main
import modelnet_classification.config as Config

# device = list(sys.argv)[1]
device = "cuda"

c: Config = Config
# output / test
c.fitting_test_data_store_at_epoch = 2000

# network size
c.layers = [Config.Layer(8, 1, 32),
            Config.Layer(16, 1, 16),
            Config.Layer(32, 1, 8),
            Config.Layer(-1, 1, -1)]

c.mnist_n_kernel_components = 5

# other network settings
c.bias_type = c.BIAS_TYPE_NONE
c.bn_place = c.BN_PLACE_AFTER_RELU
c.convolution_config.dropout = 0.1

# performance
c.batch_size = 21
c.num_dataloader_workers = 24

# https://stackoverflow.com/questions/38634988/check-if-program-runs-in-debug-mode
if getattr(sys, 'gettrace', None) is not None and getattr(sys, 'gettrace')():
    # running in debugger
    c.num_dataloader_workers = 0

main.experiment(device=device, n_epochs=200, desc_string=f"fpsmax64_2_drp{int(c.convolution_config.dropout * 100)}_allCovN_weightDec_b{c.batch_size}_{Config.produce_name(c.layers)}", kernel_learning_rate=0.001, learn_covariances_after=0, learn_positions_after=0, log_interval=c.batch_size * 10, config=c)
