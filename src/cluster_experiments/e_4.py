import sys
import modelnet_classification.main as main
from modelnet_classification.config import Config
from modelnet_classification.config import Layer

device = list(sys.argv)[1]
# device = "cuda"

c: Config = Config()
c.bn_type = Config.BN_TYPE_ONLY_COVARIANCE
c.log_tensorboard_renderings = False

# network size
c.layers = [Layer(8, 2.0, 32),
            Layer(16, 2.5, 16),
            Layer(-1, 3.0, -1)]

main.experiment(device=device, desc_string=c.produce_description(), config=c)
