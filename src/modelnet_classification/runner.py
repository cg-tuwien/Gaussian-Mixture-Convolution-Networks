import modelnet_classification.main as main
from modelnet_classification.config import Config
from modelnet_classification.config import Layer

# device = list(sys.argv)[1]
device = "cuda"

c: Config = Config(n_classes=40)

# network size
c.layers = [Layer(8, 2.5, 32),
            Layer(16, 2.5, 16),
            Layer(-1, 2.5, -1)]

main.experiment(device=device, desc_string=c.produce_description(), config=c)
