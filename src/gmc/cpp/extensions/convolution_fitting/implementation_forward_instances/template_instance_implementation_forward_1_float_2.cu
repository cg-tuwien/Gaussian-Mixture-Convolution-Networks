#include "convolution_fitting/implementation_forward.h"
#include <utility>

namespace convolution_fitting {
template ForwardOutput forward_impl_t<1, float, 2>(const torch::Tensor& data, const torch::Tensor& kernels, const Config& config);
} // namespace convolution_fitting
