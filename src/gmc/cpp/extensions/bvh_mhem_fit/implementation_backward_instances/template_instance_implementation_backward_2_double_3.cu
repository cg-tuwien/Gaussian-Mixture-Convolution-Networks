#ifndef GPE_ONLY_FLOAT
#ifndef GPE_ONLY_2D
#include "bvh_mhem_fit/implementation_backward.h"

namespace bvh_mhem_fit {
template torch::Tensor backward_impl_t<2, double, 3>(torch::Tensor grad, const ForwardOutput& forward_out, const BvhMhemFitConfig& config);
} // namespace bvh_mhem_fit
#endif // GPE_ONLY_2D
#endif // GPE_ONLY_FLOAT
