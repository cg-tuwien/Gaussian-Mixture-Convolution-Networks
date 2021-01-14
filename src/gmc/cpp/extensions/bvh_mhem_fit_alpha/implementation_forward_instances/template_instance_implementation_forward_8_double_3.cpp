#ifndef GPE_LIMIT_N_REDUCTION
#ifndef GPE_ONLY_FLOAT
#ifndef GPE_ONLY_2D
#include "bvh_mhem_fit_alpha/implementation_forward.h"

namespace bvh_mhem_fit_alpha {
template ForwardOutput forward_impl_t<8, double, 3>(torch::Tensor mixture, const Config& config);
} // namespace bvh_mhem_fit_alpha
#endif // GPE_ONLY_2D
#endif // GPE_ONLY_FLOAT
#endif // GPE_LIMIT_N_REDUCTION
