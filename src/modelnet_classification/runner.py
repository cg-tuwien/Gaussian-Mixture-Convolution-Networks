import modelnet_classification.main as main
from modelnet_classification.config import Config
from modelnet_classification.config import Layer

# device = list(sys.argv)[1]
device = "cuda"

c: Config = Config()

# network size
c.layers = [Layer(8, 1, 32),
            Layer(16, 1, 16),
            Layer(32, 1, 8),
            Layer(-1, 1, -1)]

main.experiment(device=device, desc_string=c.produce_description(), config=c)
