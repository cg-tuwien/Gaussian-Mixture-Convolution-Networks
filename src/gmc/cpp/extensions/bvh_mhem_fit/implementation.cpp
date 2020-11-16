#include "bvh_mhem_fit/implementation.h"
#include "bvh_mhem_fit/implementation_forward_template.h"
#include "mixture.h"

// todo:
// - when fitting 1024 components (more than input), mse doesn't go to 0, although the rendering looks good.
// - in collect_result, run a new fitting with the most important node to fill up the remaining gaussian slots
// - kl-divergenece filter: at least one should pass (select from clustering), otherwise we loose mass
// - check integration mass again (after kl-div filter)
// - debug 2nd and 3rd layer: why are we lossing these important gaussians, especially when N_REDUCE is larger (8)?

namespace bvh_mhem_fit {
namespace  {

template<int REDUCTION_N, typename scalar_t>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> dispatch_dim(at::Tensor mixture, const BvhMhemFitConfig& config, unsigned n_components_target, int n_dims) {
    switch (n_dims) {
    case 2:
        return forward_impl_t<2, float, 2>(mixture, config, n_components_target);
    case 3:
        return forward_impl_t<2, float, 3>(mixture, config, n_components_target);
    default:
        std::cout << "unsupported mixture.scalar_type()" << std::endl;
        exit(1);
    }
}

template<int REDUCTION_N>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> dispatch_dim_and_scalar_type(at::Tensor mixture, const BvhMhemFitConfig& config, unsigned n_components_target, int n_dims, torch::ScalarType scalar_type) {
    switch (scalar_type) {
    case torch::ScalarType::Float:
        return dispatch_dim<2, float>(mixture, config, n_components_target, n_dims);
    case torch::ScalarType::Double:
        return dispatch_dim<2, double>(mixture, config, n_components_target, n_dims);
    default:
        std::cout << "unsupported mixture.scalar_type()" << std::endl;
        exit(1);
    }
}



}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl(at::Tensor mixture, const BvhMhemFitConfig& config, unsigned n_components_target) {
    auto n_dims = gpe::n_dimensions(mixture);
    auto scalar_type = mixture.scalar_type();

    switch (config.reduction_n) {
    case 2:
        return dispatch_dim_and_scalar_type<2>(mixture, config, n_components_target, n_dims, scalar_type);
    case 4:
        return dispatch_dim_and_scalar_type<4>(mixture, config, n_components_target, n_dims, scalar_type);
    case 8:
        return dispatch_dim_and_scalar_type<8>(mixture, config, n_components_target, n_dims, scalar_type);
    case 16:
        return dispatch_dim_and_scalar_type<16>(mixture, config, n_components_target, n_dims, scalar_type);
    default:
        std::cout << "invalid BvhMhemFitConfig::reduction_n" << std::endl;
        exit(1);
    }
}

} // namespace bvh_mhem_fit

