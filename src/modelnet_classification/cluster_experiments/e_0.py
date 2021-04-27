import sys
import modelnet_classification.main as main
from modelnet_classification.config import Config
from gmc.model import Layer, Config as ModelConfig

import cluster_experiments.pc_fit as pcfit

# device = list(sys.argv)[1]
device = "cuda"

tmp_gmm_base_path = "/scratch/acelarek/gmms/e0"

fitconf = pcfit.Config(n_gaussians=128, eps=0.00001, gengmm_path=tmp_gmm_base_path)
pcfit.fit(fitconf)

c: Config = Config(gmms_fitting=fitconf.name, gengmm_path=tmp_gmm_base_path, n_classes=40)
c.model.bn_type = ModelConfig.BN_TYPE_COVARIANCE_STD
c.model.convolution_config.dropout = 0.0
c.model.dataDropout = 0.0
c.log_tensorboard_renderings = False
c.n_epochs = 160

# network size
c.model.layers = [Layer(12, 2.5, 48),
                  Layer(24, 2.5, 24),
                  Layer(48, 2.5, 12),
                  Layer(40, 2.5, -1)]
# c.model.mlp = (-1, 40)

main.experiment(device=device, desc_string=f"{fitconf.name}_{c.produce_description()}", config=c)
