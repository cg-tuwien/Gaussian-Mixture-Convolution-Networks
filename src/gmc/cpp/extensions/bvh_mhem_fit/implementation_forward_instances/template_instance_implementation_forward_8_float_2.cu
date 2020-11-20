#include "bvh_mhem_fit/implementation_forward.h"

namespace bvh_mhem_fit {
#ifndef GPE_LIMIT_N_REDUCTION
template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl_t<8, float, 2>(at::Tensor mixture, const BvhMhemFitConfig& config, unsigned n_components_target);
#endif
} // namespace bvh_mhem_fit
