#include "bvh_mhem_fit/implementation_backward.h"

namespace bvh_mhem_fit {
#ifndef GPE_LIMIT_N_REDUCTION
template torch::Tensor backward_impl_t<16, float, 3>(torch::Tensor grad, const ForwardOutput& forward_out, const BvhMhemFitConfig& config);
#endif
} // namespace bvh_mhem_fit
