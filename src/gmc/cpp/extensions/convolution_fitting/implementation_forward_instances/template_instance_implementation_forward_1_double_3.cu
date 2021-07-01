#ifndef GPE_ONLY_FLOAT
#ifndef GPE_ONLY_2D
#include "convolution_fitting/implementation_forward.h"
#include <utility>

namespace convolution_fitting {
template ForwardOutput forward_impl_t<1, double, 3>(const torch::Tensor& data, const torch::Tensor& kernels, const Config& config);
} // namespace convolution_fitting
#endif // GPE_ONLY_2D
#endif // GPE_ONLY_FLOAT
