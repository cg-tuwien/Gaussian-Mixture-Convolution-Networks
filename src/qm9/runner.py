import sys
import qm9.main as main
from qm9.config import Config
from qm9.config import Layer

# device = list(sys.argv)[1]
device = "cuda"

c: Config = Config(inference_on='U')
c.bn_type = Config.BN_TYPE_ONLY_COVARIANCE
c.bn_place = Config.BN_PLACE_NOWHERE
c.convolution_config.dropout = 0.0
c.dataDropout = 0.0
# c.log_tensorboard_renderings = False
c.n_epochs = 10
# c.training_end_index = 10000
# c.validation_start_index = 10000
# c.validation_end_index = 10200
c.batch_size = 25
c.log_interval = 2500
c.kernel_learning_rate = 0.005

# network size

# c.layers = [Layer(8, 0.5, 32),
#             Layer(16, 0.5, 32),
#             Layer(32, 0.5, -1)]
# c.mlp = (-1, 256, -1, 1)
# main.experiment(device=device, desc_string=f"qm9_{c.produce_description()}", config=c)
#
# c.layers = [Layer(8, 1.0, 32),
#             Layer(16, 1.0, 32),
#             Layer(32, 1.0, -1)]
# c.mlp = (-1, 256, -1, 1)
# main.experiment(device=device, desc_string=f"qm9_{c.produce_description()}", config=c)
#
# c.layers = [Layer(8, 1.5, 32),
#             Layer(16, 1.5, 32),
#             Layer(32, 1.5, -1)]
# c.mlp = (-1, 256, -1, 1)
# main.experiment(device=device, desc_string=f"qm9_{c.produce_description()}", config=c)
#
# c.layers = [Layer(8, 0.5, 32),
#             Layer(16, 0.5, 32),
#             Layer(32, 0.5, -1)]
# c.mlp = (-1, 1)
# main.experiment(device=device, desc_string=f"qm9_{c.produce_description()}", config=c)
#
# c.layers = [Layer(8, 1.0, 32),
#             Layer(16, 1.0, 32),
#             Layer(32, 1.0, -1)]
# c.mlp = (-1, 1)
# main.experiment(device=device, desc_string=f"qm9_{c.produce_description()}", config=c)
#
# c.mlp_dropout = 0.0
# c.layers = [Layer(8, 1.0, 32),
#             Layer(16, 1.0, 32),
#             Layer(32, 1.0, -1)]
# c.mlp = (-1, 1)
# main.experiment(device=device, desc_string=f"qm9_{c.produce_description()}", config=c)

c.layers = [Layer(16, 1.0, 8),
            Layer(32, 1.0, 8),
            Layer(64, 1.0, -1)]
c.mlp = (-1, 1)
main.experiment(device=device, desc_string=f"qm9_{c.produce_description()}", config=c)

c.layers = [Layer(8, 1.0, 16),
            Layer(16, 1.0, 16),
            Layer(32, 1.0, -1)]
c.mlp = (1, )
main.experiment(device=device, desc_string=f"qm9_{c.produce_description()}", config=c)

c.layers = [Layer(8, 1.0, 16),
            Layer(16, 1.0, 16),
            Layer(32, 1.0, -1)]
c.mlp = (-1, 1, )
main.experiment(device=device, desc_string=f"qm9_{c.produce_description()}", config=c)

c.layers = [Layer(8, 1.0, 8),
            Layer(16, 1.0, 8),
            Layer(32, 1.0, -1)]
c.mlp = (-1, 1)
main.experiment(device=device, desc_string=f"qm9_{c.produce_description()}", config=c)

c.layers = [Layer(16, 1.0, 8),
            Layer(32, 1.0, 8),
            Layer(64, 1.0, -1)]
c.mlp = (-1, 1)
main.experiment(device=device, desc_string=f"qm9_{c.produce_description()}", config=c)

c.layers = [Layer(8, 1.0, 8),
            Layer(16, 1.0, 8),
            Layer(32, 1.0, -1)]
c.mlp = (1, )
main.experiment(device=device, desc_string=f"qm9_{c.produce_description()}", config=c)

c.layers = [Layer(16, 1.0, 8),
            Layer(32, 1.0, 8),
            Layer(64, 1.0, -1)]
c.mlp = (1, )
main.experiment(device=device, desc_string=f"qm9_{c.produce_description()}", config=c)

c.layers = [Layer(16, 1.0, 16),
            Layer(32, 1.0, 16),
            Layer(64, 1.0, -1)]
c.mlp = (-1, 1)
main.experiment(device=device, desc_string=f"qm9_{c.produce_description()}", config=c)

#
# c.layers = [Layer(8, 1.5, 32),
#             Layer(16, 1.5, 32),
#             Layer(32, 1.5, -1)]
# c.mlp = (-1, 1)
# main.experiment(device=device, desc_string=f"qm9_{c.produce_description()}", config=c)
#
# c.layers = [Layer(15, 0.5, 32),
#             Layer(30, 0.5, 32),
#             Layer(1, 0.5, -1)]
# c.mlp = None
# main.experiment(device=device, desc_string=f"qm9_{c.produce_description()}", config=c)
#
# c.layers = [Layer(15, 1.0, 32),
#             Layer(30, 1.0, 32),
#             Layer(1, 1.0, -1)]
# c.mlp = None
# main.experiment(device=device, desc_string=f"qm9_{c.produce_description()}", config=c)
#
# c.layers = [Layer(15, 1.5, 32),
#             Layer(30, 1.5, 32),
#             Layer(1, 1.5, -1)]
# c.mlp = None
# main.experiment(device=device, desc_string=f"qm9_{c.produce_description()}", config=c)
#
# c.layers = [Layer(8, 0.5, 32),
#             Layer(16, 0.5, 32),
#             Layer(32, 0.5, 16),
#             Layer(1, 0.5, -1)]
# c.mlp = None
# main.experiment(device=device, desc_string=f"qm9_{c.produce_description()}", config=c)
#
# c.layers = [Layer(8, 0.75, 32),
#             Layer(16, 0.75, 32),
#             Layer(32, 0.75, 16),
#             Layer(1, 0.75, -1)]
# c.mlp = None
# main.experiment(device=device, desc_string=f"qm9_{c.produce_description()}", config=c)
#
# c.layers = [Layer(8, 1.0, 32),
#             Layer(16, 1.0, 32),
#             Layer(32, 1.0, 16),
#             Layer(1, 1.0, -1)]
# c.mlp = None
# main.experiment(device=device, desc_string=f"qm9_{c.produce_description()}", config=c)
