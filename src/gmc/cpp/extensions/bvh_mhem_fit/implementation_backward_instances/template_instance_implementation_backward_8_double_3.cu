#include "bvh_mhem_fit/implementation_backward.h"

namespace bvh_mhem_fit {
#ifndef GPE_LIMIT_N_REDUCTION
#ifndef GPE_ONLY_FLOAT
template torch::Tensor backward_impl_t<8, double, 3>(torch::Tensor grad, const ForwardOutput& forward_out, const BvhMhemFitConfig& config);
#endif // GPE_ONLY_FLOAT
#endif // GPE_LIMIT_N_REDUCTION
} // namespace bvh_mhem_fit
