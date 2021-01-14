#ifndef GPE_BVH_MHEM_FIT_ALPHA_CONFIG_H
#define GPE_BVH_MHEM_FIT_ALPHA_CONFIG_H

#include "lbvh/Config.h"

namespace bvh_mhem_fit_alpha {

struct Config {
    const int reduction_n = 4;
    lbvh::Config bvh_config = {};
    float em_kl_div_threshold = 2.0f;
    unsigned n_components_fitting = 32;
};

}
#endif // GPE_BVH_MHEM_FIT_ALPHA_CONFIG_H
