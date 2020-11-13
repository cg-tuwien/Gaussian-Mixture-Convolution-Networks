#include <tuple>
#include <torch/types.h>
#include "bvh_mhem_fit/BvhMhemFitConfig.h"

namespace bvh_mhem_fit {

template<int REDUCTION_N = 4>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl_t(at::Tensor mixture, const BvhMhemFitConfig& config, unsigned n_components_target = 32);

} // namespace bvh_mhem_fit

