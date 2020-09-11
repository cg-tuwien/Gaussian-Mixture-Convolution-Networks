#include <algorithm>
#include <chrono>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/script.h>
#include <torch/nn/functional.h>

#include <glm/glm.hpp>

#include "common.h"
#include "lbvh/aabb.h"
#include "lbvh/bvh.h"
#include "math/symeig.h"

#ifndef __CUDACC__
constexpr dim3 blockIdx;
constexpr dim3 blockDim;
constexpr dim3 threadIdx;
using std::min;
using std::max;
#endif

template<int N_DIMS, typename scalar_t>
struct Gaussian {
    scalar_t weight;
    glm::vec<N_DIMS, scalar_t> position;
    glm::mat<N_DIMS, N_DIMS, scalar_t> covariance;
};

template<int N_DIMS, typename scalar_t>
struct aabb_getter
{
    __device__
    lbvh::aabb<scalar_t> operator()(const Gaussian<N_DIMS, scalar_t>& f) const noexcept
    {
        lbvh::aabb<scalar_t> retval;
        retval.upper = f;
        retval.lower = f;
        return retval;
    }
};

torch::Tensor cuda_bvh_forward_impl(const at::Tensor& mixture, const at::Tensor& xes) {
    using namespace torch::indexing;
    namespace F = torch::nn::functional;
    using build_lbvh = lbvh::bvh<float, Gaussian<2, float>>;
    auto n = gpe::check_input_and_get_ns(mixture, xes);

    torch::Tensor sum = torch::zeros({n.batch, n.layers, n.xes}, torch::dtype(mixture.dtype()).device(mixture.device()));

    TORCH_CHECK(mixture.device().is_cuda(), "mixture must be a CUDA tensor");
    TORCH_CHECK(n.batch * n.layers < 65535, "n_batch x n_layers must be smaller than 65535 for CUDA");
    TORCH_CHECK(n.xes < 65535, "number of xes must be smaller than 65535 for CUDA");

    TORCH_CHECK(n.dims == 2, "atm only 2d gaussians");
    TORCH_CHECK(mixture.dtype() == caffe2::TypeMeta::Make<float>(), "atm only float");

    torch::Tensor aabbs;
    {
        constexpr float threshold = 0.01f;
        // double scalar = sqrt(-2 * log(threshold / m_amplitude));

        torch::Tensor factors = -2 * torch::log(threshold / gpe::weights(mixture));
        factors = factors.where(factors > 0, torch::zeros({1, 1, 1}, factors.device()));
        factors = torch::sqrt(factors);

        torch::Tensor covs = gpe::covariances(mixture).inverse();
        torch::Tensor eigenvalues;
        torch::Tensor eigenvectors;
//        std::cout << "covs.sizes()=" << covs.sizes() << std::endl;
        auto start = std::chrono::steady_clock::now();

        /// TODO: torch::symeig is too slow in pytorch version 1.6.
        std::tie(eigenvalues, eigenvectors) = gpe::symeig(covs);
        auto end = std::chrono::steady_clock::now();
        /*
         * eigenvectors is a tensor of [*, *, *, d, d], where d is the dimensionality (2 or 3)
         * the eigenvectors are in the rows of that d * d matrix.
         */
//        std::cout << "eigenvalues.sizes()=" << eigenvalues.sizes() << std::endl;
//        std::cout << "eigenvalues.unsqueeze(-1).sizes()=" << eigenvalues.unsqueeze(-1).sizes() << std::endl;
//        std::cout << "eigenvectors.sizes()=" << eigenvectors.sizes() << std::endl;
        eigenvalues = sqrt(eigenvalues);
        eigenvectors = eigenvalues.unsqueeze(-1) * eigenvectors;

        auto ellipsoidM = factors.unsqueeze(-1).unsqueeze(-1) * eigenvectors;
//        std::cout << "ellipsoidM.sizes()=" << ellipsoidM.sizes() << std::endl;

        // https://stackoverflow.com/a/24112864/4032670
        // https://members.loria.fr/SHornus/ellipsoid-bbox.html
        // we take the norm over the eigenvectors, that is analogous to simon fraiss' code in gmvis/core/Gaussian.cpp
        auto delta = torch::norm(ellipsoidM, 2, {-2});
        auto centroid = gpe::positions(mixture);
        auto upper = centroid + delta;
        auto lower = centroid - delta;

        // bring that thing into a format that can be read by our lbvh builder
        upper = F::pad(upper, F::PadFuncOptions({0, 4-n.dims}));
        lower = F::pad(lower, F::PadFuncOptions({0, 4-n.dims}));
        aabbs = torch::cat({upper, lower}, -1).contiguous();
//        std::cout << "mixture: " << mixture << std::endl;
//        std::cout << "aabbs: " << aabbs << std::endl;

        std::cout << "elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << "ms\n";
    }

    for (uint batch_id = 0; batch_id < n.batch; ++batch_id) {
        for (uint layer_id = 0; layer_id < n.layers; ++layer_id) {
            torch::Tensor current_mixture = mixture[batch_id][layer_id];
            TORCH_CHECK(current_mixture.is_contiguous(), "mixtures must be contiguous");
            auto mixture_begin = static_cast<Gaussian<2, float>*>(current_mixture.data_ptr());
            auto mixture_end = mixture_begin + n.components;
            auto aabbs_begin = static_cast<lbvh::aabb<float>*>(aabbs.data_ptr());

            auto bvh = build_lbvh(mixture_begin, mixture_end, aabbs_begin);

        }
    }


    return sum;
}

//std::vector<torch::Tensor> cuda_parallel_backward_impl(torch::Tensor grad_output, torch::Tensor mixture, torch::Tensor xes, bool requires_grad_mixture, bool requires_grad_xes) {
//    gpe::check_mixture(mixture);
//    auto n = gpe::check_input_and_get_ns(mixture, xes);

//    TORCH_CHECK(mixture.device().is_cuda(), "mixture must be a CUDA tensor")
//    TORCH_CHECK(grad_output.device().is_cuda(), "grad_output must be a CUDA tensor");
//    TORCH_CHECK(grad_output.dim() == 3, "grad_output has wrong number of dimensions");
//    TORCH_CHECK(grad_output.size(0) == n.batch, "grad_output has wrong batch dimension");
//    TORCH_CHECK(grad_output.size(1) == n.layers, "grad_output has wrong layer dimension");
//    TORCH_CHECK(grad_output.size(2) == n.xes, "grad_output has wrong xes dimension");
//    TORCH_CHECK(grad_output.dtype() == mixture.dtype(), "grad_output dtype does not match with mixture dtype")


//    torch::Tensor grad_mixture = torch::zeros({n.batch, n.layers, n.components, mixture.size(3)}, torch::dtype(mixture.dtype()).device(mixture.device()));
//    torch::Tensor grad_xes = torch::zeros({n.batch_xes, n.layers_xes, n.xes, n.dims}, torch::dtype(mixture.dtype()).device(mixture.device()));

//    dim3 dimBlock = dim3(128);
//    const dim3 dimGrid = dim3(n.batch * n.layers,
//                              n.xes,
//                              (n.components + dimBlock.z - 1) / dimBlock.z);
////    std::cout << "forward: dimBlock=" << dimBlock.x << "/" << dimBlock.y << "/" << dimBlock.z << ", dimGrid=" << dimGrid.x << "/" << dimGrid.y << "/" << dimGrid.z << std::endl;

//    AT_DISPATCH_FLOATING_TYPES(mixture.scalar_type(), "eval_inversed_omp_backward", ([&] {
//        auto mixture_a = mixture.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>();
//        auto xes_a = xes.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>();
//        auto grad_mixture_a = grad_mixture.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>();
//        auto grad_xes_a = grad_xes.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>();
//        auto grad_output_a = grad_output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();

//        if (n.dims == 2)
//            kernel_backward<scalar_t, 2><<<dimGrid, dimBlock>>>(mixture_a, xes_a, grad_mixture_a, grad_xes_a, grad_output_a, n, requires_grad_mixture, requires_grad_xes);
//        else
//            kernel_backward<scalar_t, 3><<<dimGrid, dimBlock>>>(mixture_a, xes_a, grad_mixture_a, grad_xes_a, grad_output_a, n, requires_grad_mixture, requires_grad_xes);
//    }));

//    return {grad_mixture, grad_xes};
//}
