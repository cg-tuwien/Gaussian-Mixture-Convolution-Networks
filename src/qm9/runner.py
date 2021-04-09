import sys
import qm9.main as main
from qm9.config import Config
from qm9.config import Layer

# device = list(sys.argv)[1]
device = "cuda"

c: Config = Config(inference_on='U')
c.bn_type = Config.BN_TYPE_ONLY_COVARIANCE
c.convolution_config.dropout = 0.0
c.dataDropout = 0.0
# c.log_tensorboard_renderings = False
c.n_epochs = 160
c.training_end_index = 2000
c.validation_start_index = 10000
c.validation_end_index = 10200

# network size
c.layers = [Layer(8, 0.5, 32),
            Layer(16, 0.5, 16),
            Layer(1, 0.5, -1)]
# c.mlp = (-1, 256, -1, 1)

main.experiment(device=device, desc_string=f"qm9_{c.produce_description()}bn", config=c)
