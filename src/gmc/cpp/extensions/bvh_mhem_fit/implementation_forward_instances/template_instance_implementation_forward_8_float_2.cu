#include "bvh_mhem_fit/implementation_forward.h"

namespace bvh_mhem_fit {
#ifndef GPE_LIMIT_N_REDUCTION
template ForwardOutput forward_impl_t<8, float, 2>(torch::Tensor mixture, const BvhMhemFitConfig& config);
#endif // GPE_LIMIT_N_REDUCTION
} // namespace bvh_mhem_fit
