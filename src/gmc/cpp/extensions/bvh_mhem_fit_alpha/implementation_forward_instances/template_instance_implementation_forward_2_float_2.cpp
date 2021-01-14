#include "bvh_mhem_fit_alpha/implementation_forward.h"

namespace bvh_mhem_fit_alpha {
template ForwardOutput forward_impl_t<2, float, 2>(torch::Tensor mixture, const Config& config);
} // namespace bvh_mhem_fit_alpha
