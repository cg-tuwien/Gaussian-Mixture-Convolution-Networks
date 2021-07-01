#include <tuple>
#include <torch/types.h>
#include "convolution/implementation.h"

namespace convolution {

template<typename scalar_t, unsigned N_DIMS>
ForwardOutput forward_impl_t(const torch::Tensor& data, const torch::Tensor& kernels);
extern template ForwardOutput forward_impl_t<float, 2>(const torch::Tensor& data, const torch::Tensor& kernels);
extern template ForwardOutput forward_impl_t<double, 2>(const torch::Tensor& data, const torch::Tensor& kernels);
extern template ForwardOutput forward_impl_t<float, 3>(const torch::Tensor& data, const torch::Tensor& kernels);
extern template ForwardOutput forward_impl_t<double, 3>(const torch::Tensor& data, const torch::Tensor& kernels);

template<typename scalar_t, unsigned N_DIMS>
std::pair<torch::Tensor, torch::Tensor> backward_impl_t(const torch::Tensor& grad, const ForwardOutput& forward_out);
extern template std::pair<torch::Tensor, torch::Tensor> backward_impl_t<float, 2>(const torch::Tensor& grad, const ForwardOutput& forward_out);
extern template std::pair<torch::Tensor, torch::Tensor> backward_impl_t<double, 2>(const torch::Tensor& grad, const ForwardOutput& forward_out);
extern template std::pair<torch::Tensor, torch::Tensor> backward_impl_t<float, 3>(const torch::Tensor& grad, const ForwardOutput& forward_out);
extern template std::pair<torch::Tensor, torch::Tensor> backward_impl_t<double, 3>(const torch::Tensor& grad, const ForwardOutput& forward_out);

} // namespace bvh_mhem_fit

