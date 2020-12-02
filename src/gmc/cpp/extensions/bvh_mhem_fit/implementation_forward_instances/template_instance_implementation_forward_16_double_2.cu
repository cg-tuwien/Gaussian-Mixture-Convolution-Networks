#include "bvh_mhem_fit/implementation_forward.h"

namespace bvh_mhem_fit {
#ifndef GPE_LIMIT_N_REDUCTION
#ifndef GPE_ONLY_FLOAT
template ForwardOutput forward_impl_t<16, double, 2>(torch::Tensor mixture, const BvhMhemFitConfig& config);
#endif // GPE_ONLY_FLOAT
#endif // GPE_LIMIT_N_REDUCTION
} // namespace bvh_mhem_fit
