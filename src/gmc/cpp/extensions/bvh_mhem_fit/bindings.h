#ifndef BVH_MHEM_FIT_BINDINGS
#define BVH_MHEM_FIT_BINDINGS

#include <vector>

#include <torch/types.h>

#include "BvhMhemFitConfig.h"


std::vector<torch::Tensor> bvh_mhem_fit_forward(const torch::Tensor& mixture, const BvhMhemFitConfig& config);

torch::Tensor bvh_mhem_fit_backward(const torch::Tensor& grad,
                                    const torch::Tensor& fitting_mixture,
                                    const torch::Tensor& target_mixture,
                                    const torch::Tensor& bvh_mixture_inversed, const torch::Tensor& bvh_nodes, const torch::Tensor& bvh_aabbs, const torch::Tensor& bvh_attribs,
                                    const BvhMhemFitConfig& config);

#endif
