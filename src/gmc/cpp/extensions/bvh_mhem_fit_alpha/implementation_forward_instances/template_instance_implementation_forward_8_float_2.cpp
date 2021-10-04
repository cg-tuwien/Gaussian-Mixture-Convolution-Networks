#ifndef GPE_LIMIT_N_REDUCTION
#include "bvh_mhem_fit_alpha/implementation_forward.h"

namespace bvh_mhem_fit_alpha {
template ForwardOutput forward_impl_t<8, float, 2>(torch::Tensor mixture, const Config& config);
} // namespace bvh_mhem_fit_alpha
#endif // GPE_LIMIT_N_REDUCTION
