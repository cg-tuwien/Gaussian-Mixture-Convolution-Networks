#ifndef BVHMHEMFITCONFIG_H
#define BVHMHEMFITCONFIG_H

#include "lbvh/Config.h"

struct BvhMhemFitConfig {
    const int reduction_n = 4;
    lbvh::Config bvh_config = {};

    enum class FitInitialDisparityMethod { CentroidDistance, Likelihood, KLDivergence } fit_initial_disparity_method = FitInitialDisparityMethod::KLDivergence;
    enum class FitInitialClusterMergeMethod { Average, AverageCorrected, MaxWeight, MaxIntegral } fit_initial_cluster_merge_method = FitInitialClusterMergeMethod::Average;
    float em_kl_div_threshold = 2.0f;
};

#endif // BVHMHEMFITCONFIG_H
