#ifndef GPE_ONLY_FLOAT
#include "bvh_mhem_fit/implementation_forward.h"

namespace bvh_mhem_fit {
template ForwardOutput forward_impl_t<2, double, 2>(torch::Tensor mixture, const BvhMhemFitConfig& config);
} // namespace bvh_mhem_fit
#endif // GPE_ONLY_FLOAT
