#ifndef GPE_ONLY_FLOAT
#include "convolution/implementation_backward.h"
#include <utility>

namespace convolution {
template std::pair<torch::Tensor, torch::Tensor> backward_impl_t<double, 2>(const torch::Tensor& grad, const ForwardOutput& forward_out);
} // namespace convolution
#endif // GPE_ONLY_FLOAT
