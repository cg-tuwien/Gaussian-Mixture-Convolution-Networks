#ifndef BVH_MHEM_FIT_IMPLEMENTATION
#define BVH_MHEM_FIT_IMPLEMENTATION
#include <tuple>
#include <torch/script.h>

namespace convolution {

torch::Tensor forward_impl(const at::Tensor& data, const at::Tensor& kernels);

std::pair<at::Tensor, at::Tensor> backward_impl(const torch::Tensor& grad, const torch::Tensor& data, const torch::Tensor& kernels);

}
#endif
