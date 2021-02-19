import sys
import modelnet_classification.main as main
from modelnet_classification.config import Config
from modelnet_classification.config import Layer

import cluster_experiments.pc_fit as pcfit

# device = list(sys.argv)[1]
device = "cuda"

tmp_gmm_base_path = "/home/madam/Documents/work/tuw/gmc_net/data/modelnet/gmms/tmp"

fitconf = pcfit.Config(n_gaussians=64, eps=0.00001, gengmm_path=tmp_gmm_base_path)
pcfit.fit(fitconf)

c: Config = Config(gmms_fitting=fitconf.name, gengmm_path=tmp_gmm_base_path)
c.bn_type = Config.BN_TYPE_ONLY_COVARIANCE
c.log_tensorboard_renderings = False

# network size
c.layers = [Layer(8, 2.5, 32),
            Layer(16, 2.5, 16),
            Layer(-1, 2.5, -1)]

main.experiment(device=device, desc_string=f"{fitconf.name}_{c.produce_description()}", config=c)

# network size
c.layers = [Layer(8, 2.5, 64),
            Layer(16, 2.5, 32),
            Layer(-1, 2.5, -1)]

main.experiment(device=device, desc_string=f"{fitconf.name}_{c.produce_description()}", config=c)

# network size
c.layers = [Layer(8, 2.5, 64),
            Layer(16, 2.5, 32),
            Layer(32, 2.5, 16),
            Layer(-1, 2.5, -1)]

main.experiment(device=device, desc_string=f"{fitconf.name}_{c.produce_description()}", config=c)

# network size
c.layers = [Layer(8, 1.5, 64),
            Layer(16, 1.5, 32),
            Layer(32, 1.5, 16),
            Layer(-1, 1.5, -1)]

main.experiment(device=device, desc_string=f"{fitconf.name}_{c.produce_description()}", config=c)


# em and eckart fitting

c: Config = Config(gmms_fitting="EckSP64", gengmm_path=tmp_gmm_base_path)
c.bn_type = Config.BN_TYPE_ONLY_COVARIANCE
c.log_tensorboard_renderings = False

# network size
c.layers = [Layer(8, 2.5, 32),
            Layer(16, 2.5, 16),
            Layer(-1, 2.5, -1)]

main.experiment(device=device, desc_string=f"EckSP64_{c.produce_description()}", config=c)

# network size
c.layers = [Layer(8, 1.5, 32),
            Layer(16, 1.5, 16),
            Layer(-1, 1.5, -1)]

main.experiment(device=device, desc_string=f"EckSP64_{c.produce_description()}", config=c)

# network size
c.layers = [Layer(8, 2.5, 32),
            Layer(16, 2.5, 16),
            Layer(32, 2.5, 8),
            Layer(-1, 2.5, -1)]

main.experiment(device=device, desc_string=f"EckSP64_{c.produce_description()}", config=c)