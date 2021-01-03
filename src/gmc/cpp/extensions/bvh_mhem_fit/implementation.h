#ifndef BVH_MHEM_FIT_IMPLEMENTATION
#define BVH_MHEM_FIT_IMPLEMENTATION
#include <tuple>
#include <torch/script.h>

#include "bvh_mhem_fit/Config.h"


namespace bvh_mhem_fit {
struct ForwardOutput {
    torch::Tensor fitting;
    torch::Tensor target;
    torch::Tensor bvh_mixture;
    torch::Tensor bvh_nodes;
    torch::Tensor bvh_aabbs;
    torch::Tensor bvh_attributes;

    ForwardOutput clone() {
        return {fitting.clone(), target.clone(), bvh_mixture.clone(), bvh_nodes.clone(), bvh_aabbs.clone(), bvh_attributes.clone()};
    }
};

ForwardOutput forward_impl(at::Tensor mixture, const Config& config);

torch::Tensor backward_impl(torch::Tensor grad, const ForwardOutput& forward_out, const Config& config);

}
#endif
