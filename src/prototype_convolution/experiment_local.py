import sys
import cluster_experiments.conf as cluster
import prototype_convolution.config

# device = list(sys.argv)[1]
device = "cuda"

gmcn_config: prototype_convolution.config = cluster.default_gmcn_config()
gmcn_config.bias_type = gmcn_config.BIAS_TYPE_NONE
gmcn_config.bn_place = gmcn_config.BN_PLACE_BEFORE_GMC


cluster.run_with(device, "bvhFit_noBias_bnBeforeGmc", gmcn_config)
