#ifndef GPE_ONLY_2D
#include "convolution/implementation_forward.h"
#include <utility>

namespace convolution {
template torch::Tensor forward_impl_t<float, 3>(const torch::Tensor& data, const torch::Tensor& kernels);
} // namespace convolution
#endif // GPE_ONLY_2D
