import sys
import modelnet_classification.main as main
from modelnet_classification.config import Config
from gmc.model import Layer, Config as ModelConfig

import cluster_experiments.pc_fit as pcfit

# device = list(sys.argv)[1]
device = "cuda"

tmp_gmm_base_path = None

fitconf = pcfit.Config(n_gaussians=128, eps=0.00001, gengmm_path=tmp_gmm_base_path)
# pcfit.fit(fitconf)

c: Config = Config(gmms_fitting=fitconf.name, gengmm_path=tmp_gmm_base_path, n_classes=10)
c.model.bn_type = ModelConfig.BN_TYPE_COVARIANCE
c.log_tensorboard_renderings = False

# network size
c.model.layers = [Layer(16, 2.5, 32),
                  Layer(32, 2.5, 16),
                  Layer(64, 2.5, 8),
                  Layer(10, 2.5, -1)]
main.experiment(device=device, desc_string=f"{fitconf.name}_{c.produce_description()}", config=c)
