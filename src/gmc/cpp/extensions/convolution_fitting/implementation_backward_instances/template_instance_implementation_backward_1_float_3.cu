#ifndef GPE_ONLY_2D
#include "convolution_fitting/implementation_backward.h"
#include <utility>

namespace convolution_fitting {
template std::pair<torch::Tensor, torch::Tensor> backward_impl_t<1, float, 3>(const torch::Tensor& grad, const ForwardOutput& forward_out, const Config& config);
} // namespace convolution_fitting
#endif // GPE_ONLY_2D
