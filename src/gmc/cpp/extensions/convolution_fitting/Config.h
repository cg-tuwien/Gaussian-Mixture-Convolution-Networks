#ifndef CONVOLUTION_FITTING_CONFIG_H
#define CONVOLUTION_FITTING_CONFIG_H

#include "lbvh/Config.h"

namespace convolution_fitting {
struct Config {
    int reduction_n = 1;
    lbvh::Config bvh_config = {};
    float em_kl_div_threshold = 0.5f;
    unsigned n_components_fitting = 32;
};

namespace constants {
constexpr unsigned n_bits_for_sign = 1;
constexpr unsigned n_bits_for_morton_code = 30;
constexpr unsigned n_bits_for_channel_in_id = 12;
constexpr unsigned n_bits_for_data_id = 17;
constexpr unsigned n_bits_for_kernel_id = 4;

static_assert (n_bits_for_sign + n_bits_for_morton_code + n_bits_for_channel_in_id + n_bits_for_data_id + n_bits_for_kernel_id == 64, "number of code bits does not add up to 64");
}

} // namespace bvh_mhem_fit

#endif // BVHMHEMFITCONFIG_H
