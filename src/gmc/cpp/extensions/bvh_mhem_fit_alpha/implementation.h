#ifndef GPE_BVH_MHEM_FIT_ALPHA_IMPLEMENTATION
#define GPE_BVH_MHEM_FIT_ALPHA_IMPLEMENTATION
#include <tuple>
#include <torch/script.h>

#include "Config.h"


namespace bvh_mhem_fit_alpha {
struct ForwardOutput {
    torch::Tensor fitting;
    torch::Tensor target;
    torch::Tensor bvh_mixture;
    torch::Tensor bvh_nodes;
    torch::Tensor bvh_aabbs;
    torch::Tensor bvh_attributes;
};


ForwardOutput forward_impl(at::Tensor mixture, const Config& config);

torch::Tensor backward_impl(torch::Tensor grad, const ForwardOutput& forward_out, const Config& config);

}
#endif
