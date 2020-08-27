import sys
import cluster_experiments.conf as cluster
import prototype_convolution.config
import prototype_convolution.fitting

# device = list(sys.argv)[1]
device = "cuda"

gmcn_config: prototype_convolution.config = cluster.default_gmcn_config()
gmcn_config.bias_type = gmcn_config.BIAS_TYPE_NONE
gmcn_config.bn_place = gmcn_config.BN_PLACE_BEFORE_GMC
gmcn_config.fitting_config.representative_select_mode = prototype_convolution.fitting.Config.REPRESENTATIVE_SELECT_MODE_RANDOM_TOP

cluster.run_with(device, "fitRndTop_biasNone_bnAfterGmc", gmcn_config)
