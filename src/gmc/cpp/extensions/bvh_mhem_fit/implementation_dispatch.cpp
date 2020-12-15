#include "bvh_mhem_fit/implementation.h"
#include "bvh_mhem_fit/implementation_template_externs.h"
#include "util/mixture.h"

namespace bvh_mhem_fit {
namespace  {

template<int REDUCTION_N, typename scalar_t>
ForwardOutput dispatch_forward_dim(at::Tensor mixture, const BvhMhemFitConfig& config, int n_dims) {
    switch (n_dims) {
    case 2:
        return forward_impl_t<REDUCTION_N, scalar_t, 2>(mixture, config);
#ifndef GPE_ONLY_2D
    case 3:
        return forward_impl_t<REDUCTION_N, scalar_t, 3>(mixture, config);
#endif // GPE_ONLY_2D
    default:
        std::cout << "unsupported mixture.scalar_type()" << std::endl;
        exit(1);
    }
}

template<int REDUCTION_N>
ForwardOutput dispatch_forward_dim_and_scalar_type(at::Tensor mixture, const BvhMhemFitConfig& config, int n_dims, torch::ScalarType scalar_type) {
    switch (scalar_type) {
    case torch::ScalarType::Float:
        return dispatch_forward_dim<REDUCTION_N, float>(mixture, config, n_dims);
#ifndef GPE_ONLY_FLOAT
    case torch::ScalarType::Double:
        return dispatch_forward_dim<REDUCTION_N, double>(mixture, config, n_dims);
#endif // GPE_ONLY_FLOAT
    default:
        std::cout << "unsupported mixture.scalar_type()" << std::endl;
        exit(1);
    }
}

template<int REDUCTION_N, typename scalar_t>
torch::Tensor dispatch_backward_dim(at::Tensor grad, const ForwardOutput& forward_out, const BvhMhemFitConfig& config, int n_dims) {
    switch (n_dims) {
    case 2:
        return backward_impl_t<REDUCTION_N, scalar_t, 2>(grad, forward_out, config);
#ifndef GPE_ONLY_2D
    case 3:
        return backward_impl_t<REDUCTION_N, scalar_t, 3>(grad, forward_out, config);
#endif // GPE_ONLY_2D
    default:
        std::cout << "unsupported mixture.scalar_type()" << std::endl;
        exit(1);
    }
}

template<int REDUCTION_N>
torch::Tensor dispatch_backward_dim_and_scalar_type(at::Tensor grad, const ForwardOutput& forward_out, const BvhMhemFitConfig& config, int n_dims, torch::ScalarType scalar_type) {
    switch (scalar_type) {
    case torch::ScalarType::Float:
        return dispatch_backward_dim<REDUCTION_N, float>(grad, forward_out, config, n_dims);
#ifndef GPE_ONLY_FLOAT
    case torch::ScalarType::Double:
        return dispatch_backward_dim<REDUCTION_N, double>(grad, forward_out, config, n_dims);
#endif // GPE_ONLY_FLOAT
    default:
        std::cout << "unsupported mixture.scalar_type()" << std::endl;
        exit(1);
    }
}

}

ForwardOutput forward_impl(at::Tensor mixture, const BvhMhemFitConfig& config) {
    auto n_dims = gpe::n_dimensions(mixture);
    auto scalar_type = mixture.scalar_type();

    switch (config.reduction_n) {
    case 2:
        return dispatch_forward_dim_and_scalar_type<2>(mixture, config, n_dims, scalar_type);
    case 4:
        return dispatch_forward_dim_and_scalar_type<4>(mixture, config, n_dims, scalar_type);
#ifndef GPE_LIMIT_N_REDUCTION
    case 8:
        return dispatch_forward_dim_and_scalar_type<8>(mixture, config, n_dims, scalar_type);
    case 16:
        return dispatch_forward_dim_and_scalar_type<16>(mixture, config, n_dims, scalar_type);
#endif
    default:
        std::cout << "invalid BvhMhemFitConfig::reduction_n" << std::endl;
        exit(1);
    }
}

torch::Tensor backward_impl(at::Tensor grad, const ForwardOutput& forward_out, const BvhMhemFitConfig& config) {
    auto n_dims = gpe::n_dimensions(grad);
    auto scalar_type = grad.scalar_type();

    switch (config.reduction_n) {
    case 2:
        return dispatch_backward_dim_and_scalar_type<2>(grad, forward_out, config, n_dims, scalar_type);
    case 4:
        return dispatch_backward_dim_and_scalar_type<4>(grad, forward_out, config, n_dims, scalar_type);
#ifndef GPE_LIMIT_N_REDUCTION
    case 8:
        return dispatch_backward_dim_and_scalar_type<8>(grad, forward_out, config, n_dims, scalar_type);
    case 16:
        return dispatch_backward_dim_and_scalar_type<16>(grad, forward_out, config, n_dims, scalar_type);
#endif
    default:
        std::cout << "invalid BvhMhemFitConfig::reduction_n" << std::endl;
        exit(1);
    }
}

} // namespace bvh_mhem_fit

