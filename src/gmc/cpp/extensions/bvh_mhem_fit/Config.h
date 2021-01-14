#ifndef BVHMHEMFITCONFIG_H
#define BVHMHEMFITCONFIG_H

#include "lbvh/Config.h"

namespace bvh_mhem_fit {
struct Config {
    int reduction_n = 4;
    lbvh::Config bvh_config = {};
    float em_kl_div_threshold = 0.5f;
    unsigned n_components_fitting = 32;
};

} // namespace bvh_mhem_fit

#endif // BVHMHEMFITCONFIG_H
