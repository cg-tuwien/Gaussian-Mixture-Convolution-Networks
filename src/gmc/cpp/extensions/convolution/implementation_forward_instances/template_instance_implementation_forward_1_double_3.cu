#ifndef GPE_ONLY_FLOAT
#ifndef GPE_ONLY_2D
#include "convolution/implementation_forward.h"
#include <utility>

namespace convolution {
template ForwardOutput forward_impl_t<1, double, 3>(const torch::Tensor& data, const torch::Tensor& kernels, const Config& config);
} // namespace convolution
#endif // GPE_ONLY_2D
#endif // GPE_ONLY_FLOAT
