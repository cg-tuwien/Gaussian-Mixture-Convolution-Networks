import sys
import modelnet_classification.main as main
from modelnet_classification.config import Config
from gmc.model import Layer, Config as ModelConfig

import cluster_experiments.pc_fit as pcfit

# device = list(sys.argv)[1]
device = "cuda"

tmp_gmm_base_path = None

# fitconf = pcfit.Config(n_gaussians=128, eps=0.00001, gengmm_path=tmp_gmm_base_path)
# pcfit.fit(fitconf)

# c: Config = Config(gmms_fitting=fitconf.name, gengmm_path=tmp_gmm_base_path, n_classes=10)
c: Config = Config(gmms_fitting="fpsmax64", gengmm_path=tmp_gmm_base_path, n_classes=10)
c.model.bn_type = ModelConfig.BN_TYPE_COVARIANCE
c.model.dropout = 0.0
c.log_tensorboard_renderings = False
c.n_epochs = 52

# network size
c.model.layers = [Layer(8, 2.5, 4),
                  # Layer(16, 2.5, 4),
                  # Layer(32, 2.5, -1),
                  # Layer(10, 2.5, -1),
                  ]
# c.model.mlp = (16, 32, 10, )

# main.experiment(device=device, desc_string=f"{fitconf.name}_{c.produce_description()}", config=c)
main.experiment(device=device, desc_string=f"fpsmax64_{c.produce_description()}", config=c)