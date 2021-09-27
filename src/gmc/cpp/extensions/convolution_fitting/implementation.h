#ifndef CONVOLUTION_FITTING_IMPLEMENTATION
#define CONVOLUTION_FITTING_IMPLEMENTATION
#include <tuple>
#include <torch/script.h>

#include "convolution_fitting/Config.h"


namespace convolution_fitting {
struct ForwardOutput {
    torch::Tensor fitting;
    torch::Tensor cached_pos_covs;
    torch::Tensor nodesobjs;
    torch::Tensor fitting_subtrees;
};

ForwardOutput forward_impl(const at::Tensor& data, const at::Tensor& kernels, const Config& config);

std::pair<at::Tensor, at::Tensor> backward_impl(torch::Tensor grad, const at::Tensor& data, const at::Tensor& kernels, const ForwardOutput& forward_out, const Config& config);

}
#endif
