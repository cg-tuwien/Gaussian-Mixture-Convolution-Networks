#include <tuple>
#include <torch/types.h>
#include "bvh_mhem_fit/BvhMhemFitConfig.h"

namespace bvh_mhem_fit {

template<int REDUCTION_N = 4, typename scalar_t, unsigned N_DIMS>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl_t(at::Tensor mixture, const BvhMhemFitConfig& config, unsigned n_components_target = 32);

extern template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl_t<2,  float, 2>(at::Tensor mixture, const BvhMhemFitConfig& config, unsigned n_components_target);
extern template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl_t<4,  float, 2>(at::Tensor mixture, const BvhMhemFitConfig& config, unsigned n_components_target);
extern template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl_t<8,  float, 2>(at::Tensor mixture, const BvhMhemFitConfig& config, unsigned n_components_target);
extern template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl_t<16, float, 2>(at::Tensor mixture, const BvhMhemFitConfig& config, unsigned n_components_target);

extern template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl_t<2,  double, 2>(at::Tensor mixture, const BvhMhemFitConfig& config, unsigned n_components_target);
extern template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl_t<4,  double, 2>(at::Tensor mixture, const BvhMhemFitConfig& config, unsigned n_components_target);
extern template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl_t<8,  double, 2>(at::Tensor mixture, const BvhMhemFitConfig& config, unsigned n_components_target);
extern template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl_t<16, double, 2>(at::Tensor mixture, const BvhMhemFitConfig& config, unsigned n_components_target);

extern template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl_t<2,  float, 3>(at::Tensor mixture, const BvhMhemFitConfig& config, unsigned n_components_target);
extern template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl_t<4,  float, 3>(at::Tensor mixture, const BvhMhemFitConfig& config, unsigned n_components_target);
extern template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl_t<8,  float, 3>(at::Tensor mixture, const BvhMhemFitConfig& config, unsigned n_components_target);
extern template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl_t<16, float, 3>(at::Tensor mixture, const BvhMhemFitConfig& config, unsigned n_components_target);

extern template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl_t<2,  double, 3>(at::Tensor mixture, const BvhMhemFitConfig& config, unsigned n_components_target);
extern template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl_t<4,  double, 3>(at::Tensor mixture, const BvhMhemFitConfig& config, unsigned n_components_target);
extern template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl_t<8,  double, 3>(at::Tensor mixture, const BvhMhemFitConfig& config, unsigned n_components_target);
extern template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl_t<16, double, 3>(at::Tensor mixture, const BvhMhemFitConfig& config, unsigned n_components_target);

} // namespace bvh_mhem_fit

