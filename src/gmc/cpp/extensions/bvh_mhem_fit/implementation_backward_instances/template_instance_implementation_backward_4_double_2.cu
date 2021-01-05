#ifndef GPE_ONLY_FLOAT
#include "bvh_mhem_fit/implementation_backward.h"

namespace bvh_mhem_fit {
template torch::Tensor backward_impl_t<4, double, 2>(torch::Tensor grad, const ForwardOutput& forward_out, const Config& config);
} // namespace bvh_mhem_fit
#endif // GPE_ONLY_FLOAT
