#ifndef CONVOLUTION_FITTING_BINDINGS
#define CONVOLUTION_FITTING_BINDINGS

#include <vector>

#include <torch/types.h>



std::vector<torch::Tensor> convolution_fitting_forward(torch::Tensor data, torch::Tensor kernels, int n_components_fitting, int reduction_n);

//std::pair<at::Tensor, at::Tensor> convolution_fitting_backward(const torch::Tensor& grad,
//                                    const torch::Tensor& fitting_mixture,
//                                    const torch::Tensor& target_mixture, const torch::Tensor& bvh_nodes, const torch::Tensor& bvh_attribs, int n_components_fitting, int reduction_n);

#endif
