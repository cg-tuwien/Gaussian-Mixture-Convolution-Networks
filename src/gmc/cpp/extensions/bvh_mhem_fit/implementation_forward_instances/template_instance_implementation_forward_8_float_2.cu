#ifndef GPE_LIMIT_N_REDUCTION
#include "bvh_mhem_fit/implementation_forward.h"

namespace bvh_mhem_fit {
template ForwardOutput forward_impl_t<8, float, 2>(torch::Tensor mixture, const Config& config);
} // namespace bvh_mhem_fit
#endif // GPE_LIMIT_N_REDUCTION
