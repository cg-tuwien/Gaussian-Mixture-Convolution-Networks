#include <tuple>
#include <torch/types.h>
#include "convolution_fitting/Config.h"
#include "convolution_fitting/implementation.h"

namespace convolution_fitting {

template<int REDUCTION_N = 4, typename scalar_t, unsigned N_DIMS>
ForwardOutput forward_impl_t(const torch::Tensor& data, const torch::Tensor& kernels, const Config& configForwardOutput);

extern template ForwardOutput forward_impl_t<1,  float, 2>(const torch::Tensor& data, const torch::Tensor& kernels, const Config& configForwardOutput);
//extern template ForwardOutput forward_impl_t<2,  float, 2>(const torch::Tensor& data, const torch::Tensor& kernels, const Config& configForwardOutput);
//extern template ForwardOutput forward_impl_t<4,  float, 2>(const torch::Tensor& data, const torch::Tensor& kernels, const Config& configForwardOutput);
//extern template ForwardOutput forward_impl_t<8,  float, 2>(const torch::Tensor& data, const torch::Tensor& kernels, const Config& configForwardOutput);

extern template ForwardOutput forward_impl_t<1,  double, 2>(const torch::Tensor& data, const torch::Tensor& kernels, const Config& configForwardOutput);
//extern template ForwardOutput forward_impl_t<2,  double, 2>(const torch::Tensor& data, const torch::Tensor& kernels, const Config& configForwardOutput);
//extern template ForwardOutput forward_impl_t<4,  double, 2>(const torch::Tensor& data, const torch::Tensor& kernels, const Config& configForwardOutput);
//extern template ForwardOutput forward_impl_t<8,  double, 2>(const torch::Tensor& data, const torch::Tensor& kernels, const Config& configForwardOutput);

extern template ForwardOutput forward_impl_t<1,  float, 3>(const torch::Tensor& data, const torch::Tensor& kernels, const Config& configForwardOutput);
//extern template ForwardOutput forward_impl_t<2,  float, 3>(const torch::Tensor& data, const torch::Tensor& kernels, const Config& configForwardOutput);
//extern template ForwardOutput forward_impl_t<4,  float, 3>(const torch::Tensor& data, const torch::Tensor& kernels, const Config& configForwardOutput);
//extern template ForwardOutput forward_impl_t<8,  float, 3>(const torch::Tensor& data, const torch::Tensor& kernels, const Config& configForwardOutput);

extern template ForwardOutput forward_impl_t<1,  double, 3>(const torch::Tensor& data, const torch::Tensor& kernels, const Config& configForwardOutput);
//extern template ForwardOutput forward_impl_t<2,  double, 3>(const torch::Tensor& data, const torch::Tensor& kernels, const Config& configForwardOutput);
//extern template ForwardOutput forward_impl_t<4,  double, 3>(const torch::Tensor& data, const torch::Tensor& kernels, const Config& configForwardOutput);
//extern template ForwardOutput forward_impl_t<8,  double, 3>(const torch::Tensor& data, const torch::Tensor& kernels, const Config& configForwardOutput);

template<int REDUCTION_N = 4, typename scalar_t, unsigned N_DIMS>
std::pair<torch::Tensor, torch::Tensor> backward_impl_t(const torch::Tensor& grad, const torch::Tensor& data, const torch::Tensor& kernels, const ForwardOutput& forward_out, const Config& config);

extern template std::pair<torch::Tensor, torch::Tensor> backward_impl_t<1,  float, 2>(const torch::Tensor& grad, const torch::Tensor& data, const torch::Tensor& kernels, const ForwardOutput& forward_out, const Config& config);
//extern template std::pair<torch::Tensor, torch::Tensor> backward_impl_t<2,  float, 2>(const torch::Tensor& grad, const torch::Tensor& data, const torch::Tensor& kernels, const ForwardOutput& forward_out, const Config& config);
//extern template std::pair<torch::Tensor, torch::Tensor> backward_impl_t<4,  float, 2>(const torch::Tensor& grad, const torch::Tensor& data, const torch::Tensor& kernels, const ForwardOutput& forward_out, const Config& config);
//extern template std::pair<torch::Tensor, torch::Tensor> backward_impl_t<8,  float, 2>(const torch::Tensor& grad, const torch::Tensor& data, const torch::Tensor& kernels, const ForwardOutput& forward_out, const Config& config);

extern template std::pair<torch::Tensor, torch::Tensor> backward_impl_t<1,  double, 2>(const torch::Tensor& grad, const torch::Tensor& data, const torch::Tensor& kernels, const ForwardOutput& forward_out, const Config& config);
//extern template std::pair<torch::Tensor, torch::Tensor> backward_impl_t<2,  double, 2>(const torch::Tensor& grad, const torch::Tensor& data, const torch::Tensor& kernels, const ForwardOutput& forward_out, const Config& config);
//extern template std::pair<torch::Tensor, torch::Tensor> backward_impl_t<4,  double, 2>(const torch::Tensor& grad, const torch::Tensor& data, const torch::Tensor& kernels, const ForwardOutput& forward_out, const Config& config);
//extern template std::pair<torch::Tensor, torch::Tensor> backward_impl_t<8,  double, 2>(const torch::Tensor& grad, const torch::Tensor& data, const torch::Tensor& kernels, const ForwardOutput& forward_out, const Config& config);

extern template std::pair<torch::Tensor, torch::Tensor> backward_impl_t<1,  float, 3>(const torch::Tensor& grad, const torch::Tensor& data, const torch::Tensor& kernels, const ForwardOutput& forward_out, const Config& config);
//extern template std::pair<torch::Tensor, torch::Tensor> backward_impl_t<2,  float, 3>(const torch::Tensor& grad, const torch::Tensor& data, const torch::Tensor& kernels, const ForwardOutput& forward_out, const Config& config);
//extern template std::pair<torch::Tensor, torch::Tensor> backward_impl_t<4,  float, 3>(const torch::Tensor& grad, const torch::Tensor& data, const torch::Tensor& kernels, const ForwardOutput& forward_out, const Config& config);
//extern template std::pair<torch::Tensor, torch::Tensor> backward_impl_t<8,  float, 3>(const torch::Tensor& grad, const torch::Tensor& data, const torch::Tensor& kernels, const ForwardOutput& forward_out, const Config& config);

extern template std::pair<torch::Tensor, torch::Tensor> backward_impl_t<1,  double, 3>(const torch::Tensor& grad, const torch::Tensor& data, const torch::Tensor& kernels, const ForwardOutput& forward_out, const Config& config);
//extern template std::pair<torch::Tensor, torch::Tensor> backward_impl_t<2,  double, 3>(const torch::Tensor& grad, const torch::Tensor& data, const torch::Tensor& kernels, const ForwardOutput& forward_out, const Config& config);
//extern template std::pair<torch::Tensor, torch::Tensor> backward_impl_t<4,  double, 3>(const torch::Tensor& grad, const torch::Tensor& data, const torch::Tensor& kernels, const ForwardOutput& forward_out, const Config& config);
//extern template std::pair<torch::Tensor, torch::Tensor> backward_impl_t<8,  double, 3>(const torch::Tensor& grad, const torch::Tensor& data, const torch::Tensor& kernels, const ForwardOutput& forward_out, const Config& config);

} // namespace bvh_mhem_fit

