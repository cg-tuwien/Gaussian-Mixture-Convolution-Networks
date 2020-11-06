#ifndef BVH_MHEM_FIT_BINDINGS
#define BVH_MHEM_FIT_BINDINGS

#include <vector>
#include <algorithm>

#include <torch/extension.h>

#include "BvhMhemFitConfig.h"


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bvh_mhem_fit_forward(const torch::Tensor& mixture, const BvhMhemFitConfig& config, unsigned n_components_target);

std::tuple<torch::Tensor, torch::Tensor> bvh_mhem_fit_backward(const torch::Tensor& grad_output,
                                                               const torch::Tensor& mixture, const torch::Tensor& bvh_nodes, const torch::Tensor& aabbs,
                                                               const torch::Tensor& xes,
                                                               bool requires_grad_mixture, bool requires_grad_xes);

#endif
