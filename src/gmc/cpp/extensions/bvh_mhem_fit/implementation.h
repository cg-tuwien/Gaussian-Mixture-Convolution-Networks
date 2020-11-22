#ifndef BVH_MHEM_FIT_IMPLEMENTATION
#define BVH_MHEM_FIT_IMPLEMENTATION
#include <tuple>
#include <torch/script.h>

#include "BvhMhemFitConfig.h"


namespace bvh_mhem_fit {
struct ForwardOutput {
    torch::Tensor fitting;
    torch::Tensor target;
    torch::Tensor bvh_mixture;
    torch::Tensor bvh_nodes;
    torch::Tensor bvh_aabbs;
    torch::Tensor bvh_attributes;
};


ForwardOutput forward_impl(at::Tensor mixture, const BvhMhemFitConfig& config);

torch::Tensor backward_impl(torch::Tensor grad, const ForwardOutput& forward_out, const BvhMhemFitConfig& config);

}
#endif
