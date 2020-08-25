import sys
import prototype_convolution.config as gmcn_config
import cluster_experiments.conf as cluster

device = list(sys.argv)[1]
#device = "cuda"

gmcn_config.bias_type = gmcn_config.BIAS_TYPE_NONE
gmcn_config.bn_place = gmcn_config.BN_PLACE_NOWHERE
cluster.run_with(device, "biasNone_bnNone", gmcn_config)

gmcn_config.bias_type = gmcn_config.BIAS_TYPE_NORMAL
gmcn_config.bn_place = gmcn_config.BN_PLACE_NOWHERE
cluster.run_with(device, "biasYes_bnNone", gmcn_config)
