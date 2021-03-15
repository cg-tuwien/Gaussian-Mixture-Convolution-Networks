import modelnet_classification.main as main
from modelnet_classification.config import Config
from modelnet_classification.config import Layer

# device = list(sys.argv)[1]
device = "cuda"

c: Config = Config(n_classes=40)
c.bn_type = Config.BN_TYPE_ONLY_COVARIANCE

# network size
c.layers = [Layer(8, 2.5, 32),
            Layer(16, 2.5, -1), ]
#            Layer(40, 2.5, -1)]
c.mlp = (64, 40, )

main.experiment(device=device, desc_string=f"{c.produce_description()}bn", config=c)
