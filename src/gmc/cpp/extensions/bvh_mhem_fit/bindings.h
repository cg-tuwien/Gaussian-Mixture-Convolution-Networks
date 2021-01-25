#ifndef BVH_MHEM_FIT_BINDINGS
#define BVH_MHEM_FIT_BINDINGS

#include <vector>

#include <torch/types.h>



std::vector<torch::Tensor> bvh_mhem_fit_forward(const torch::Tensor& mixture, int n_components_fitting, int reduction_n);

torch::Tensor bvh_mhem_fit_backward(const torch::Tensor& grad,
                                    const torch::Tensor& fitting_mixture,
                                    const torch::Tensor& target_mixture, const torch::Tensor& bvh_nodes, const torch::Tensor& bvh_attribs, int n_components_fitting, int reduction_n);

#endif
