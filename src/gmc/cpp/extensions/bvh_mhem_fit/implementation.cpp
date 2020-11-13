#include "bvh_mhem_fit/implementation.h"
#include "bvh_mhem_fit/implementation_forward_template.h"

// todo:
// - when fitting 1024 components (more than input), mse doesn't go to 0, although the rendering looks good.
// - in collect_result, run a new fitting with the most important node to fill up the remaining gaussian slots
// - kl-divergenece filter: at least one should pass (select from clustering), otherwise we loose mass
// - check integration mass again (after kl-div filter)
// - debug 2nd and 3rd layer: why are we lossing these important gaussians, especially when N_REDUCE is larger (8)?

namespace bvh_mhem_fit {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl(at::Tensor mixture, const BvhMhemFitConfig& config, unsigned n_components_target) {
    switch (config.reduction_n) {
    case 2:
        return forward_impl_t<2>(mixture, config, n_components_target);
    case 4:
        return forward_impl_t<4>(mixture, config, n_components_target);
    case 8:
        return forward_impl_t<8>(mixture, config, n_components_target);
//    case 16:
//        return forward_impl_t<16>(mixture, config, n_components_target);
    default:
        std::cout << "invalid BvhMhemFitConfig::reduction_n" << std::endl;
        exit(1);

    }
}

} // namespace bvh_mhem_fit

