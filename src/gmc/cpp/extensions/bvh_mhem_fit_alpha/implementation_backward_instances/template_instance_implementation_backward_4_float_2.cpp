#include "bvh_mhem_fit_alpha/implementation_backward.h"

namespace bvh_mhem_fit_alpha {
template torch::Tensor backward_impl_t<4, float, 2>(torch::Tensor grad, const ForwardOutput& forward_out, const Config& config);
} // namespace bvh_mhem_fit_alpha
