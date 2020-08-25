import sys
import prototype_convolution.config as gmcn_config
import cluster_experiments.conf as cluster

device = list(sys.argv)[1]
#device = "cuda"

gmcn_config.bias_type = gmcn_config.BIAS_TYPE_NORMAL
gmcn_config.bn_place = gmcn_config.BN_PLACE_BEFORE_GMC
cluster.run_with(device, "biasYes_bnBeforeGmc", gmcn_config)

gmcn_config.bias_type = gmcn_config.BIAS_TYPE_NORMAL
gmcn_config.bn_place = gmcn_config.BN_PLACE_BEFORE_GMC
cluster.run_with(device, "biasYes_bnAfterGmc", gmcn_config)


