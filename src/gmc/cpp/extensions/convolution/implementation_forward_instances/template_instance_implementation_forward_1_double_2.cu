#ifndef GPE_ONLY_FLOAT
#include "convolution/implementation_forward.h"
#include <utility>

namespace convolution {
template ForwardOutput forward_impl_t<double, 2>(const torch::Tensor& data, const torch::Tensor& kernels);
} // namespace convolution
#endif // GPE_ONLY_FLOAT
