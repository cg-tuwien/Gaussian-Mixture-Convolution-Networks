import sys
import modelnet_classification.main as main
from modelnet_classification.config import Config
from modelnet_classification.config import Layer

device = list(sys.argv)[1]
# device = "cuda"

c: Config = Config()
c.bn_type = Config.BN_TYPE_COVARIANCE_INTEGRAL

# network size
c.layers = [Layer(8, 1, 32),
            Layer(16, 1, 16),
            Layer(32, 1, 8),
            Layer(-1, 1, -1)]

main.experiment(device=device, desc_string=c.produce_description(), config=c)

# network size
c.layers = [Layer(8, 1.5, 32),
            Layer(16, 1.5, 16),
            Layer(32, 1.5, 8),
            Layer(-1, 1.5, -1)]

main.experiment(device=device, desc_string=c.produce_description(), config=c)

# network size
c.layers = [Layer(8, 2, 32),
            Layer(16, 2, 16),
            Layer(32, 2, 8),
            Layer(-1, 2, -1)]

main.experiment(device=device, desc_string=c.produce_description(), config=c)

