import sys
import mnist_classification.main as main
from mnist_classification.config import Config, Layer

# device = list(sys.argv)[1]
device = "cuda"

c: Config = Config()
c.bn_type = Config.BN_TYPE_COVARIANCE_STD
c.bn_place = Config.BN_PLACE_AFTER_RELU
c.convolution_config.dropout = 0.0
c.dataDropout = 0.0
# c.log_tensorboard_renderings = False
c.n_epochs = 10
c.batch_size = 50
c.log_interval = 1000

# network size
c.layers = [Layer(8, 1.0, 32),
            Layer(16, 1.0, 16),
            Layer(32, 1.0, 8),
            Layer(64, 1.0, 4),
            Layer(10, 1.0, -1)]
# c.mlp = (-1, 10)

main.experiment(device=device, desc_string=f"mnist_{c.produce_description()}", config=c)
