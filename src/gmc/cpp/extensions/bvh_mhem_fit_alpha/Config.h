#ifndef GPE_BVH_MHEM_FIT_ALPHA_CONFIG_H
#define GPE_BVH_MHEM_FIT_ALPHA_CONFIG_H

#include "lbvh/Config.h"

namespace bvh_mhem_fit_alpha {

struct Config {
    const int reduction_n = 4;

    lbvh::Config bvh_config = {};

    // other methods removed because they didn't perform well
    enum class FitInitialDisparityMethod { CentroidDistance/*, Likelihood, KLDivergence*/ } fit_initial_disparity_method = FitInitialDisparityMethod::CentroidDistance;
    enum class FitInitialClusterMergeMethod { /*Average, AverageCorrected, */MaxWeight/*, MaxIntegral */} fit_initial_cluster_merge_method = FitInitialClusterMergeMethod::MaxWeight;

    float em_kl_div_threshold = 2.0f;

    unsigned n_components_fitting = 32;
};

}
#endif // GPE_BVH_MHEM_FIT_ALPHA_CONFIG_H
