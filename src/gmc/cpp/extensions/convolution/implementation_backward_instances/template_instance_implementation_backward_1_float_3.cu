#ifndef GPE_ONLY_2D
#include "convolution/implementation_backward.h"
#include <utility>

namespace convolution {
template std::pair<torch::Tensor, torch::Tensor> backward_impl_t<float, 3>(const torch::Tensor& grad, const ForwardOutput& forward_out);
} // namespace convolution
#endif // GPE_ONLY_2D
