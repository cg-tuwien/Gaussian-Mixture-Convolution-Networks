#include "bvh_mhem_fit/implementation_forward.h"

namespace bvh_mhem_fit {
#ifndef GPE_ONLY_FLOAT
template ForwardOutput forward_impl_t<2, double, 3>(torch::Tensor mixture, const BvhMhemFitConfig& config);
#endif // GPE_ONLY_FLOAT
} // namespace bvh_mhem_fit
