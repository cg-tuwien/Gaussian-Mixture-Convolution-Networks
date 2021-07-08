#ifndef CONVOLUTION_BINDINGS
#define CONVOLUTION_BINDINGS

#include <vector>

#include <torch/types.h>



torch::Tensor convolution_forward(torch::Tensor data, torch::Tensor kernels);

std::pair<torch::Tensor, torch::Tensor> convolution_backward(torch::Tensor grad, torch::Tensor data, torch::Tensor kernels);

#endif
