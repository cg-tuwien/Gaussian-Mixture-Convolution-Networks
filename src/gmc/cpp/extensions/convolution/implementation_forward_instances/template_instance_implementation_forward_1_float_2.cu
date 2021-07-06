#include "convolution/implementation_forward.h"
#include <utility>

namespace convolution {
template torch::Tensor forward_impl_t<float, 2>(const torch::Tensor& data, const torch::Tensor& kernels);
} // namespace convolution
