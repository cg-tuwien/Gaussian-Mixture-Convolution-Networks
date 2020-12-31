#ifndef GPE_ONLY_FLOAT
#include "bvh_mhem_fit_alpha/implementation_forward.h"

namespace bvh_mhem_fit_alpha {
template ForwardOutput forward_impl_t<2, double, 2>(torch::Tensor mixture, const Config& config);
} // namespace bvh_mhem_fit_alpha
#endif // GPE_ONLY_FLOAT
