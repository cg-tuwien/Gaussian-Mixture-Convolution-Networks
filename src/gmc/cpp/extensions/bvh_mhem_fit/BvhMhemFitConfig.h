#ifndef BVHMHEMFITCONFIG_H
#define BVHMHEMFITCONFIG_H

#include "lbvh/Config.h"

struct BvhMhemFitConfig {
    const int reduction_n = 4;
    lbvh::Config bvh_config = {};
};

#endif // BVHMHEMFITCONFIG_H
