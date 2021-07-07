#ifndef GPE_ONLY_FLOAT
#include "convolution/implementation_backward.h"
#include <utility>

namespace convolution {
template std::pair<torch::Tensor, torch::Tensor> backward_impl_t<double, 2>(const torch::Tensor& grad, const torch::Tensor& data, const torch::Tensor& kernels);
} // namespace convolution
#endif // GPE_ONLY_FLOAT
