#ifndef GPE_ONLY_FLOAT
#ifndef GPE_ONLY_2D
#include "bvh_mhem_fit/implementation_forward.h"

namespace bvh_mhem_fit {
template ForwardOutput forward_impl_t<4, double, 3>(torch::Tensor mixture, const BvhMhemFitConfig& config);
} // namespace bvh_mhem_fit
#endif // GPE_ONLY_2D
#endif // GPE_ONLY_FLOAT
