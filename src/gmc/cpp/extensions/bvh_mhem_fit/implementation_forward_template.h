#include <tuple>
#include <torch/types.h>
#include "bvh_mhem_fit/BvhMhemFitConfig.h"

namespace bvh_mhem_fit {

template<int REDUCTION_N = 4>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl_t(at::Tensor mixture, const BvhMhemFitConfig& config, unsigned n_components_target = 32);

extern template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl_t<2>(at::Tensor mixture, const BvhMhemFitConfig& config, unsigned n_components_target);
extern template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl_t<4>(at::Tensor mixture, const BvhMhemFitConfig& config, unsigned n_components_target);
extern template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl_t<8>(at::Tensor mixture, const BvhMhemFitConfig& config, unsigned n_components_target);
extern template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl_t<16>(at::Tensor mixture, const BvhMhemFitConfig& config, unsigned n_components_target);

} // namespace bvh_mhem_fit

