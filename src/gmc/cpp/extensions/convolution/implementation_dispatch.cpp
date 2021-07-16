#include "convolution/implementation.h"
#include "convolution/implementation_template_externs.h"
#include "util/mixture.h"

namespace convolution {
namespace  {

template<typename scalar_t>
torch::Tensor dispatch_forward_dim(const torch::Tensor& data, const torch::Tensor& kernels, int n_dims) {
    switch (n_dims) {
    case 2:
        return forward_impl_t<scalar_t, 2>(data, kernels);
#ifndef GPE_ONLY_2D
    case 3:
        return forward_impl_t<scalar_t, 3>(data, kernels);
#endif // GPE_ONLY_2D
    default:
        std::cout << "unsupported mixture.scalar_type()" << std::endl;
        exit(1);
    }
}

torch::Tensor dispatch_forward_dim_and_scalar_type(const torch::Tensor& data, const torch::Tensor& kernels, int n_dims, torch::ScalarType scalar_type) {
    switch (scalar_type) {
    case torch::ScalarType::Float:
        return dispatch_forward_dim<float>(data, kernels, n_dims);
#ifndef GPE_ONLY_FLOAT
    case torch::ScalarType::Double:
        return dispatch_forward_dim<double>(data, kernels, n_dims);
#endif // GPE_ONLY_FLOAT
    default:
        std::cout << "unsupported mixture.scalar_type()" << std::endl;
        exit(1);
    }
}

template<typename scalar_t>
std::pair<torch::Tensor, torch::Tensor> dispatch_backward_dim(torch::Tensor grad, const torch::Tensor& data, const torch::Tensor& kernels, int n_dims) {
    switch (n_dims) {
    case 2:
        return backward_impl_t<scalar_t, 2>(grad, data, kernels);
#ifndef GPE_ONLY_2D
    case 3:
        return backward_impl_t<scalar_t, 3>(grad, data, kernels);
#endif // GPE_ONLY_2D
    default:
        std::cout << "unsupported mixture.scalar_type()" << std::endl;
        exit(1);
    }
}

std::pair<torch::Tensor, torch::Tensor> dispatch_backward_dim_and_scalar_type(torch::Tensor grad, const torch::Tensor& data, const torch::Tensor& kernels, int n_dims, torch::ScalarType scalar_type) {
    switch (scalar_type) {
    case torch::ScalarType::Float:
        return dispatch_backward_dim<float>(grad, data, kernels, n_dims);
#ifndef GPE_ONLY_FLOAT
    case torch::ScalarType::Double:
        return dispatch_backward_dim<double>(grad, data, kernels, n_dims);
#endif // GPE_ONLY_FLOAT
    default:
        std::cout << "unsupported mixture.scalar_type()" << std::endl;
        exit(1);
    }
}

}

torch::Tensor forward_impl(const torch::Tensor& data, const torch::Tensor& kernels) {
    auto n_dims = gpe::n_dimensions(data);
    auto scalar_type = data.scalar_type();
    return dispatch_forward_dim_and_scalar_type(data, kernels, n_dims, scalar_type);
}

std::pair<torch::Tensor, torch::Tensor> backward_impl(const torch::Tensor& grad, const torch::Tensor& data, const torch::Tensor& kernels) {
    auto n_dims = gpe::n_dimensions(grad);
    auto scalar_type = grad.scalar_type();
    return dispatch_backward_dim_and_scalar_type(grad, data, kernels, n_dims, scalar_type);
}

} // namespace bvh_mhem_fit

