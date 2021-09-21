#include "evaluate_inversed/implementations.h"

//#include <torch/extension.h>
#include <torch/script.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <algorithm>

#include "common.h"
#include "cuda_qt_creator_definitinos.h"
#include "hacked_accessor.h"
#include "util/scalar.h"
#include "parallel_start.h"
#include "util/gaussian_mixture.h"
#include "util/mixture.h"
#include "util/grad/gaussian.h"

namespace {

template <typename scalar_t, int DIMS>
__host__ __device__
void forward(const dim3& gpe_gridDim, const dim3& gpe_blockDim,
             const dim3& gpe_blockIdx, const dim3& gpe_threadIdx,
             const gpe::PackedTensorAccessor32<scalar_t, 4> mixture_a,
             const gpe::PackedTensorAccessor32<scalar_t, 4> xes_a,
             gpe::PackedTensorAccessor32<scalar_t, 3> sum_a,
             const gpe::MixtureAndXesNs n) {
    GPE_UNUSED(gpe_gridDim)
    const auto batch_index = int(gpe_blockIdx.z);
    const auto layer_index = int(gpe_blockIdx.y);
    const auto batch_xes_index = gpe::min(batch_index, n.batch_xes - 1);
    const auto layer_xes_index = gpe::min(layer_index, n.layers_xes - 1);
    const auto xes_index = int(gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x);

    if (xes_index >= n.xes)
        return;

    for (int component_index = 0; component_index < n.components; ++component_index) {
        const auto xa = xes_a[batch_xes_index];
        const auto xb = xa[layer_xes_index];
        const auto xc = xb[xes_index];
        const auto& xd = xc[0];
        const auto& x_pos = gpe::vec<DIMS>(xd);

        const auto& c_weight = gpe::weight(mixture_a[batch_index][layer_index][component_index]);
        const auto& c_pos = gpe::position<DIMS>(mixture_a[batch_index][layer_index][component_index]);
        const auto& c_cov = gpe::covariance<DIMS>(mixture_a[batch_index][layer_index][component_index]);
        const auto w = gpe::evaluate_inversed(gpe::Gaussian<DIMS, scalar_t>{c_weight, c_pos, c_cov}, x_pos);

        sum_a[batch_index][layer_index][xes_index] += w;
    }
}



}

at::Tensor parallel_forward_impl(const torch::Tensor& mixture, const torch::Tensor& xes) {
    using namespace torch::indexing;
    auto n = gpe::check_input_and_get_ns(mixture, xes);

    torch::Tensor sum = torch::zeros({n.batch, n.layers, n.xes}, torch::dtype(mixture.dtype()).device(mixture.device()));

    TORCH_CHECK(mixture.device() == xes.device(), "mixture and xes must be on the same device")
    TORCH_CHECK(n.batch * n.layers < 65535, "n_batch x n_layers must be smaller than 65535 for CUDA")


    dim3 dimBlock = dim3(128, 1, 1);
    const dim3 dimGrid = dim3((uint(n.xes) + dimBlock.x - 1) / dimBlock.x,
                              uint(n.layers),
                              uint(n.batch));
    //    std::cout << "forward: dimBlock=" << dimBlock.x << "/" << dimBlock.y << "/" << dimBlock.z << ", dimGrid=" << dimGrid.x << "/" << dimGrid.y << "/" << dimGrid.z << std::endl;


    AT_DISPATCH_FLOATING_TYPES(mixture.scalar_type(), "cuda_parallel_forward_impl", ([&] {
                                   auto sum_a = gpe::accessor<scalar_t, 3>(sum);
                                   auto mixture_a = gpe::accessor<scalar_t, 4>(mixture);
                                   const auto xes_a = gpe::accessor<scalar_t, 4>(xes);

                                   if (n.dims == 2) {
                                       auto fun = [mixture_a, xes_a, sum_a, n] __host__ __device__
                                           (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
                                               forward<scalar_t, 2>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx, mixture_a, xes_a, sum_a, n);
                                           };
                                       gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(mixture), dimGrid, dimBlock, fun);
                                   }
                                   else {
                                       auto fun = [mixture_a, xes_a, sum_a, n] __host__ __device__
                                           (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
                                               forward<scalar_t, 3>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx, mixture_a, xes_a, sum_a, n);
                                           };
                                       gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(mixture), dimGrid, dimBlock, fun);
                                   }

                               }));
    return sum;
}

template<typename scalar_t, unsigned N_DIMS>
std::tuple<torch::Tensor, torch::Tensor> parallel_backward_impl_t(const torch::Tensor& grad_output, const torch::Tensor& mixture, const torch::Tensor& xes, bool requires_grad_mixture, bool requires_grad_xes) {
    using Gaussian = gpe::Gaussian<N_DIMS, scalar_t>;
    using Vec = typename Gaussian::pos_t;
#ifndef NDEBUG
    gpe::check_mixture(mixture);
#endif


    auto n = gpe::check_input_and_get_ns(mixture, xes);

    TORCH_CHECK(grad_output.dim() == 3, "grad_output has wrong number of dimensions")
    TORCH_CHECK(grad_output.size(0) == n.batch, "grad_output has wrong batch dimension")
    TORCH_CHECK(grad_output.size(1) == n.layers, "grad_output has wrong layer dimension")
    TORCH_CHECK(grad_output.size(2) == n.xes, "grad_output has wrong xes dimension")
    TORCH_CHECK(grad_output.dtype() == mixture.dtype(), "grad_output dtype does not match with mixture dtype")


    torch::Tensor grad_mixture = torch::zeros({n.batch, n.layers, n.components, mixture.size(3)}, torch::dtype(mixture.dtype()).device(mixture.device()));
    torch::Tensor grad_xes = torch::zeros({n.batch_xes, n.layers_xes, n.xes, n.dims}, torch::dtype(mixture.dtype()).device(mixture.device()));

    dim3 dimBlock = dim3(128, 1, 1);
    const dim3 dimGrid = dim3((uint(n.xes) + dimBlock.x - 1) / dimBlock.x,
                              uint(n.layers),
                              uint(n.batch));

    auto mixture_a = gpe::struct_accessor<Gaussian, 3>(mixture);
    auto xes_a = gpe::struct_accessor<Vec, 3>(xes);
    auto grad_mixture_a = gpe::struct_accessor<Gaussian, 3>(grad_mixture);
    auto grad_xes_a = gpe::struct_accessor<Vec, 3>(grad_xes);
    auto grad_output_a = gpe::accessor<scalar_t, 3>(grad_output);
    gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(mixture), dimGrid, dimBlock, [=] __host__ __device__
                                                  (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) mutable {
        using gradless_scalar_t = gpe::remove_grad_t<scalar_t>;
        GPE_UNUSED(gpe_gridDim)
        const auto batch_index = (gpe_blockIdx.z);
        const auto layer_index = (gpe_blockIdx.y);
        const auto batch_xes_index = gpe::min(batch_index, unsigned(n.batch_xes - 1));
        const auto layer_xes_index = gpe::min(layer_index, unsigned(n.layers_xes - 1));
        const auto xes_index = (gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x);

        if (xes_index >= unsigned(n.xes))
            return;
        //    printf("block %d/%d/%d, thread %d: batch_index=%d, layer_index=%d, component_index=%d, xes_index=%d \n",
        //           blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x,
        //           batch_index, layer_index, component_index, xes_index);

        for (unsigned component_index = 0; component_index < unsigned(n.components); ++component_index) {
            const auto& x_pos = xes_a[batch_xes_index][layer_xes_index][xes_index];
            const auto& component = mixture_a[batch_index][layer_index][component_index];
            const auto incoming_grad = grad_output_a[batch_index][layer_index][xes_index];

            Gaussian grad_component = {};
            Vec grad_x_pos = {};
            gpe::grad::evaluate_inversed(component, x_pos, &grad_component, &grad_x_pos, incoming_grad);

            if (requires_grad_xes) {
                for (int i = 0; i < int(N_DIMS); ++i) {
                    gpe::atomicAdd(&grad_xes_a[batch_xes_index][layer_xes_index][xes_index][i], grad_x_pos[i]);
                }
            }
            if (requires_grad_mixture) {
                Gaussian& grad_mixture = grad_mixture_a[batch_index][layer_index][component_index];
                gpe::atomicAdd(&grad_mixture.weight, grad_component.weight);
                for (int i = 0; i < int(N_DIMS); ++i) {
                    gpe::atomicAdd(&grad_mixture.position[i], grad_component.position[i]);
                    for (int j = 0; j < int(N_DIMS); ++j)
                        gpe::atomicAdd(&grad_mixture.covariance[i][j], grad_component.covariance[i][j]);
                }
            }
        }
    });

    return std::make_tuple(grad_mixture, grad_xes);
}


std::tuple<torch::Tensor, torch::Tensor> parallel_backward_impl(const torch::Tensor& grad_output, const torch::Tensor& mixture, const torch::Tensor& xes, bool requires_grad_mixture, bool requires_grad_xes) {
    if (gpe::n_dimensions(mixture) == 2 && mixture.scalar_type() == torch::ScalarType::Float)
        return parallel_backward_impl_t<float, 2>(grad_output, mixture, xes, requires_grad_mixture, requires_grad_xes);

    if (gpe::n_dimensions(mixture) == 2 && mixture.scalar_type() == torch::ScalarType::Double)
        return parallel_backward_impl_t<double, 2>(grad_output, mixture, xes, requires_grad_mixture, requires_grad_xes);

    if (gpe::n_dimensions(mixture) == 3 && mixture.scalar_type() == torch::ScalarType::Float)
        return parallel_backward_impl_t<float, 3>(grad_output, mixture, xes, requires_grad_mixture, requires_grad_xes);

    if (gpe::n_dimensions(mixture) == 3 && mixture.scalar_type() == torch::ScalarType::Double)
        return parallel_backward_impl_t<double, 3>(grad_output, mixture, xes, requires_grad_mixture, requires_grad_xes);

    TORCH_CHECK(false, "unsupported datatype or number of dimensions")

    return {};
}
