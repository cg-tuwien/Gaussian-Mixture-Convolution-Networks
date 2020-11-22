#include "bvh_mhem_fit/implementation_forward.h"

namespace bvh_mhem_fit {
template ForwardOutput forward_impl_t<4, float, 2>(torch::Tensor mixture, const BvhMhemFitConfig& config);
} // namespace bvh_mhem_fit
