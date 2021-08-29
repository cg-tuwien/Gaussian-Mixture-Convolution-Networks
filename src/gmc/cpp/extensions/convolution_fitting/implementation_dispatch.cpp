#include "convolution_fitting/implementation.h"
#include "convolution_fitting/implementation_template_externs.h"
#include "util/mixture.h"

namespace convolution_fitting {
namespace  {

template<int REDUCTION_N, typename scalar_t>
ForwardOutput dispatch_forward_dim(const torch::Tensor& data, const torch::Tensor& kernels, const Config& config, int n_dims) {
    switch (n_dims) {
    case 2:
        return forward_impl_t<REDUCTION_N, scalar_t, 2>(data, kernels, config);
#ifndef GPE_ONLY_2D
    case 3:
        return forward_impl_t<REDUCTION_N, scalar_t, 3>(data, kernels, config);
#endif // GPE_ONLY_2D
    default:
        std::cout << "unsupported mixture.scalar_type()" << std::endl;
        exit(1);
    }
}

template<int REDUCTION_N>
ForwardOutput dispatch_forward_dim_and_scalar_type(const torch::Tensor& data, const torch::Tensor& kernels, const Config& config, int n_dims, torch::ScalarType scalar_type) {
    switch (scalar_type) {
    case torch::ScalarType::Float:
        return dispatch_forward_dim<REDUCTION_N, float>(data, kernels, config, n_dims);
#ifndef GPE_ONLY_FLOAT
    case torch::ScalarType::Double:
        return dispatch_forward_dim<REDUCTION_N, double>(data, kernels, config, n_dims);
#endif // GPE_ONLY_FLOAT
    default:
        std::cout << "unsupported mixture.scalar_type()" << std::endl;
        exit(1);
    }
}

template<int REDUCTION_N, typename scalar_t>
std::pair<torch::Tensor, torch::Tensor> dispatch_backward_dim(torch::Tensor grad, const ForwardOutput& forward_out, const Config& config, int n_dims) {
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
std::pair<torch::Tensor, torch::Tensor> dispatch_backward_dim_and_scalar_type(torch::Tensor grad, const ForwardOutput& forward_out, const Config& config, int n_dims, torch::ScalarType scalar_type) {
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

ForwardOutput forward_impl(const torch::Tensor& data, const torch::Tensor& kernels, const Config& config) {
    auto n_dims = gpe::n_dimensions(data);
    auto scalar_type = data.scalar_type();

//    switch (config.reduction_n) {
//    case 1:
        return dispatch_forward_dim_and_scalar_type<1>(data, kernels, config, n_dims, scalar_type);
//    case 2:
//        return dispatch_forward_dim_and_scalar_type<2>(mixture, config, n_dims, scalar_type);
//    case 4:
//        return dispatch_forward_dim_and_scalar_type<4>(mixture, config, n_dims, scalar_type);
//#ifndef GPE_LIMIT_N_REDUCTION
//    case 8:
//        return dispatch_forward_dim_and_scalar_type<8>(mixture, config, n_dims, scalar_type);
//#endif
//    default:
//        std::cout << "invalid convolution_fitting::Config::reduction_n" << std::endl;
//        exit(1);
//    }
}

std::pair<torch::Tensor, torch::Tensor> backward_impl(torch::Tensor grad, const ForwardOutput& forward_out, const Config& config) {
    auto n_dims = gpe::n_dimensions(grad);
    auto scalar_type = grad.scalar_type();

//    switch (config.reduction_n) {
//    case 1:
        return dispatch_backward_dim_and_scalar_type<1>(grad, forward_out, config, n_dims, scalar_type);
//    case 2:
//        return dispatch_backward_dim_and_scalar_type<2>(grad, forward_out, config, n_dims, scalar_type);
//    case 4:
//        return dispatch_backward_dim_and_scalar_type<4>(grad, forward_out, config, n_dims, scalar_type);
//#ifndef GPE_LIMIT_N_REDUCTION
//    case 8:
//        return dispatch_backward_dim_and_scalar_type<8>(grad, forward_out, config, n_dims, scalar_type);
//#endif
//    default:
//        std::cout << "invalid convolution_fitting::Config::reduction_n" << std::endl;
//        exit(1);
//    }
}

} // namespace bvh_mhem_fit

