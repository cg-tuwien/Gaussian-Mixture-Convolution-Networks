#ifndef GPE_LIMIT_N_REDUCTION
#ifndef GPE_ONLY_FLOAT
#ifndef GPE_ONLY_2D
#include "bvh_mhem_fit_alpha/implementation_backward.h"

namespace bvh_mhem_fit_alpha {
template torch::Tensor backward_impl_t<16, double, 3>(torch::Tensor grad, const ForwardOutput& forward_out, const Config& config);
} // namespace bvh_mhem_fit_alpha
#endif // GPE_ONLY_2D
#endif // GPE_ONLY_FLOAT
#endif // GPE_LIMIT_N_REDUCTION
