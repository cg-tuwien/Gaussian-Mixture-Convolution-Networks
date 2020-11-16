#include "bvh_mhem_fit/implementation_forward.h"

namespace bvh_mhem_fit {
template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl_t<2, float, 3>(at::Tensor mixture, const BvhMhemFitConfig& config, unsigned n_components_target);
} // namespace bvh_mhem_fit
