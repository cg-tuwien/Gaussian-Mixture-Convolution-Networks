import sys
import modelnet_classification.main as main
from modelnet_classification.config import Config
from modelnet_classification.config import Layer

import cluster_experiments.pc_fit as pcfit

device = list(sys.argv)[1]
# device = "cuda"

tmp_gmm_base_path = "/scratch/acelarek/gmms"

fitconf = pcfit.Config(n_gaussians=256, eps=0.00001, gengmm_path=tmp_gmm_base_path)
pcfit.fit(fitconf)

c: Config = Config(gmms_fitting=fitconf.name, gengmm_path=tmp_gmm_base_path, n_classes=40)
c.bn_type = Config.BN_TYPE_ONLY_COVARIANCE
c.log_tensorboard_renderings = False
c.n_kernel_components = 4

# network size
c.layers = [Layer(8, 1.5, 64),
            Layer(16, 1.5, 32),
            Layer(32, 1.5, 16),
            Layer(-1, 1.5, -1)]

main.experiment(device=device, desc_string=f"{fitconf.name}_{c.produce_description()}", config=c)
