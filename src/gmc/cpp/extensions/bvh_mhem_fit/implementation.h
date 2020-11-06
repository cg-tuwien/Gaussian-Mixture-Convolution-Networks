#ifndef BVH_MHEM_FIT_IMPLEMENTATION
#define BVH_MHEM_FIT_IMPLEMENTATION
#include <tuple>
#include <torch/script.h>

#include "BvhMhemFitConfig.h"


namespace bvh_mhem_fit {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl(at::Tensor mixture, const BvhMhemFitConfig& config, unsigned n_components_target = 32);

std::tuple<torch::Tensor, torch::Tensor> backward_impl(const torch::Tensor& grad_output,
                                                       const torch::Tensor& mixture, const torch::Tensor& bvh_nodes, const torch::Tensor& aabbs,
                                                       const torch::Tensor& xes,
                                                       bool requires_grad_mixture, bool requires_grad_xes);

}
#endif
