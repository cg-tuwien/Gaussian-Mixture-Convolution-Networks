#include "bvh_mhem_fit/implementation_backward.h"

namespace bvh_mhem_fit {
#ifndef GPE_ONLY_FLOAT
template torch::Tensor backward_impl_t<4, double, 2>(torch::Tensor grad, const ForwardOutput& forward_out, const BvhMhemFitConfig& config);
#endif // GPE_ONLY_FLOAT
} // namespace bvh_mhem_fit
