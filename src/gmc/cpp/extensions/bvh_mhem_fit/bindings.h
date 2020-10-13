#include <vector>
#include <algorithm>

#include <torch/extension.h>

#ifndef BVH_MHEM_FIT_BINDINGS
#define BVH_MHEM_FIT_BINDINGS

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bvh_mhem_fit_forward(const torch::Tensor& mixture, const torch::Tensor& xes);

std::tuple<torch::Tensor, torch::Tensor> bvh_mhem_fit_backward(const torch::Tensor& grad_output,
                                                               const torch::Tensor& mixture, const torch::Tensor& bvh_nodes, const torch::Tensor& aabbs,
                                                               const torch::Tensor& xes,
                                                               bool requires_grad_mixture, bool requires_grad_xes);

#endif
