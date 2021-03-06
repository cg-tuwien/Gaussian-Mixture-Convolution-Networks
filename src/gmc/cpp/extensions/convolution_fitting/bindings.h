#ifndef CONVOLUTION_FITTING_BINDINGS
#define CONVOLUTION_FITTING_BINDINGS

#include <vector>

#include <torch/types.h>



std::vector<torch::Tensor> convolution_fitting_forward(torch::Tensor data, torch::Tensor kernels, int n_components_fitting);

std::pair<at::Tensor, at::Tensor> convolution_fitting_backward(const torch::Tensor& grad,
                                                               const torch::Tensor& data, const torch::Tensor& kernels, int n_components_fitting,
                                                               const torch::Tensor& fitting, const torch::Tensor& cached_pos_covs, const torch::Tensor& nodeobjs, const torch::Tensor& fitting_subtrees);

#endif
