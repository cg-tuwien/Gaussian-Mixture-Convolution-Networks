#include "parallel_implementation.h"

//#include <torch/extension.h>
#include <torch/script.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <algorithm>

#include "common.h"
#include "parallel_start.h"
#include "mixture.h"
#include "math/scalar.h"
#include "cuda_qt_creator_definitinos.h"

namespace {

template <typename scalar_t, int DIMS>
__host__ __device__
void forward(const dim3& gpe_gridDim, const dim3& gpe_blockDim,
             const dim3& gpe_blockIdx, const dim3& gpe_threadIdx,
             const torch::PackedTensorAccessor32<scalar_t, 4> mixture_a,
             const torch::PackedTensorAccessor32<scalar_t, 4> xes_a,
             torch::PackedTensorAccessor32<scalar_t, 3> sum_a,
             const gpe::MixtureAndXesNs n) {
    GPE_UNUSED(gpe_gridDim)
    const auto batch_index = gpe_blockIdx.z;
    const auto layer_index = gpe_blockIdx.y;
    const auto batch_xes_index = gpe::min(batch_index, n.batch_xes - 1);
    const auto layer_xes_index = gpe::min(layer_index, n.layers_xes - 1);
    const auto xes_index = gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x;

    if (xes_index >= uint(n.xes))
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
        const auto w = gpe::evaluate_gaussian(x_pos, c_weight, c_pos, c_cov);

        sum_a[batch_index][layer_index][xes_index] += w;
    }
}


template <typename scalar_t, int DIMS>
__host__ __device__
void backward(const dim3& gpe_gridDim, const dim3& gpe_blockDim,
              const dim3& gpe_blockIdx, const dim3& gpe_threadIdx,
              const torch::PackedTensorAccessor32<scalar_t, 4> mixture_a,
              const torch::PackedTensorAccessor32<scalar_t, 4> xes_a,
              torch::PackedTensorAccessor32<scalar_t, 4> grad_mixture_a,
              torch::PackedTensorAccessor32<scalar_t, 4> grad_xes_a,
              const torch::PackedTensorAccessor32<scalar_t, 3> grad_output_a,
              const gpe::MixtureAndXesNs n, bool requires_grad_mixture, bool requires_grad_xes) {
    GPE_UNUSED(gpe_gridDim)
    const auto batch_index = gpe_blockIdx.x / n.layers;
    const auto layer_index = gpe_blockIdx.x - batch_index * n.layers;
    const auto batch_xes_index = min(batch_index, n.batch_xes - 1);
    const auto layer_xes_index = min(layer_index, n.layers_xes - 1);
    const auto xes_index = gpe_blockIdx.y;
    const auto component_index = gpe_blockIdx.z * gpe_blockDim.z + gpe_threadIdx.z;

    if (component_index >= uint(n.components))
        return;
    //    printf("block %d/%d/%d, thread %d: batch_index=%d, layer_index=%d, component_index=%d, xes_index=%d \n",
    //           blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x,
    //           batch_index, layer_index, component_index, xes_index);

    auto a = xes_a[batch_xes_index][layer_xes_index][xes_index];
    const scalar_t& memory_location = a[0];
    const auto& x_pos = gpe::vec<DIMS>(memory_location);

    //    glm::vec<DIMS, scalar_t>& grad_xes = gpe::vec<DIMS>(grad_xes_a[batch_layer_index][xes_index][0]);

    //    auto& grad_c_weight = gpe::weight(grad_mixture_a[batch_layer_index][component_index]);
    //    auto& grad_c_pos = gpe::position<DIMS>(grad_mixture_a[batch_layer_index][component_index]);
    //    auto& grad_c_cov = gpe::covariance<DIMS>(grad_mixture_a[batch_layer_index][component_index]);

    const auto& c_weight = gpe::weight(mixture_a[batch_index][layer_index][component_index]);
    const auto& c_pos = gpe::position<DIMS>(mixture_a[batch_index][layer_index][component_index]);
    const auto& c_cov = gpe::covariance<DIMS>(mixture_a[batch_index][layer_index][component_index]);

    const auto t = x_pos - c_pos;
    const auto v = scalar_t(-0.5) * glm::dot(t, (c_cov * t));
    const auto exp = gpe::exp(v);
    const auto weighted_exp = c_weight * exp;
    const auto local_grad_c_pos = weighted_exp * t * c_cov;

    if (requires_grad_xes) {
        const auto grad_xes_addition = - grad_output_a[batch_index][layer_index][xes_index] * local_grad_c_pos;
        for (uint i = 0; i < DIMS; ++i) {
            gpe::atomicAdd(&grad_xes_a[batch_xes_index][layer_xes_index][xes_index][i], grad_xes_addition[i]);
        }
    }
    if (requires_grad_mixture) {
        const auto grad_c_weight_addition = exp * grad_output_a[batch_index][layer_index][xes_index];
        const auto grad_c_pos_addition = local_grad_c_pos * grad_output_a[batch_index][layer_index][xes_index];
        const auto grad_c_cov_addition = - c_weight * scalar_t(0.5) * exp * grad_output_a[batch_index][layer_index][xes_index] * glm::outerProduct(t, t);
        gpe::atomicAdd(&grad_mixture_a[batch_index][layer_index][component_index][0], grad_c_weight_addition);
        for (uint i = 0; i < DIMS; ++i) {
            gpe::atomicAdd(&grad_mixture_a[batch_index][layer_index][component_index][1 + i], grad_c_pos_addition[i]);
            for (uint j = 0; j < DIMS; ++j)
                gpe::atomicAdd(&grad_mixture_a[batch_index][layer_index][component_index][1 + DIMS + i*DIMS + j], grad_c_cov_addition[i][j]);
        }
    }
}

}

at::Tensor parallel_forward_impl(const torch::Tensor& mixture, const torch::Tensor& xes) {
    using namespace torch::indexing;
    auto n = gpe::check_input_and_get_ns(mixture, xes);

    torch::Tensor sum = torch::zeros({n.batch, n.layers, n.xes}, torch::dtype(mixture.dtype()).device(mixture.device()));

    TORCH_CHECK(mixture.device() == xes.device(), "mixture and xes must be on the same device");
    TORCH_CHECK(n.batch * n.layers < 65535, "n_batch x n_layers must be smaller than 65535 for CUDA");
    TORCH_CHECK(n.xes < 65535, "number of xes must be smaller than 65535 for CUDA");


    dim3 dimBlock = dim3(128, 1, 1);
    const dim3 dimGrid = dim3((n.xes + dimBlock.x - 1) / dimBlock.x,
                              n.layers,
                              n.batch);
    //    std::cout << "forward: dimBlock=" << dimBlock.x << "/" << dimBlock.y << "/" << dimBlock.z << ", dimGrid=" << dimGrid.x << "/" << dimGrid.y << "/" << dimGrid.z << std::endl;


    AT_DISPATCH_FLOATING_TYPES(mixture.scalar_type(), "cuda_parallel_forward_impl", ([&] {
                                   auto mixture_a = mixture.packed_accessor32<scalar_t, 4>();
                                   auto xes_a = xes.packed_accessor32<scalar_t, 4>();
                                   auto sum_a = sum.packed_accessor32<scalar_t, 3>();

                                   if (n.dims == 2) {
                                       auto fun = [mixture_a, xes_a, sum_a, n] __host__ __device__
                                           (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
                                               forward<scalar_t, 2>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx, mixture_a, xes_a, sum_a, n);
                                           };
                                       gpe::start_parallel(gpe::device(mixture), dimGrid, dimBlock, fun);
                                   }
                                   else {
                                       auto fun = [mixture_a, xes_a, sum_a, n] __host__ __device__
                                           (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
                                               forward<scalar_t, 3>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx, mixture_a, xes_a, sum_a, n);
                                           };
                                       gpe::start_parallel(gpe::device(mixture), dimGrid, dimBlock, fun);
                                   }

                                   cudaDeviceSynchronize();
                               }));
    return sum;
}

std::tuple<torch::Tensor, torch::Tensor> parallel_backward_impl(const torch::Tensor& grad_output, const torch::Tensor& mixture, const torch::Tensor& xes, bool requires_grad_mixture, bool requires_grad_xes) {
    gpe::check_mixture(mixture);
    auto n = gpe::check_input_and_get_ns(mixture, xes);

    TORCH_CHECK(grad_output.dim() == 3, "grad_output has wrong number of dimensions");
    TORCH_CHECK(grad_output.size(0) == n.batch, "grad_output has wrong batch dimension");
    TORCH_CHECK(grad_output.size(1) == n.layers, "grad_output has wrong layer dimension");
    TORCH_CHECK(grad_output.size(2) == n.xes, "grad_output has wrong xes dimension");
    TORCH_CHECK(grad_output.dtype() == mixture.dtype(), "grad_output dtype does not match with mixture dtype")


    torch::Tensor grad_mixture = torch::zeros({n.batch, n.layers, n.components, mixture.size(3)}, torch::dtype(mixture.dtype()).device(mixture.device()));
    torch::Tensor grad_xes = torch::zeros({n.batch_xes, n.layers_xes, n.xes, n.dims}, torch::dtype(mixture.dtype()).device(mixture.device()));

    dim3 dimBlock = dim3(128);
    const dim3 dimGrid = dim3(n.batch * n.layers,
                              n.xes,
                              (n.components + dimBlock.z - 1) / dimBlock.z);
    //    std::cout << "forward: dimBlock=" << dimBlock.x << "/" << dimBlock.y << "/" << dimBlock.z << ", dimGrid=" << dimGrid.x << "/" << dimGrid.y << "/" << dimGrid.z << std::endl;

    AT_DISPATCH_FLOATING_TYPES(mixture.scalar_type(), "cuda_parallel_backward_impl", ([&] {
                                   auto mixture_a = mixture.packed_accessor32<scalar_t, 4>();
                                   auto xes_a = xes.packed_accessor32<scalar_t, 4>();
                                   auto grad_mixture_a = grad_mixture.packed_accessor32<scalar_t, 4>();
                                   auto grad_xes_a = grad_xes.packed_accessor32<scalar_t, 4>();
                                   auto grad_output_a = grad_output.packed_accessor32<scalar_t, 3>();

                                   if (n.dims == 2) {
                                       auto fun = [=] __host__ __device__
                                           (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
                                               backward<scalar_t, 2>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx,
                                                                     mixture_a, xes_a,
                                                                     grad_mixture_a, grad_xes_a, grad_output_a,
                                                                     n, requires_grad_mixture, requires_grad_xes);
                                           };
                                       gpe::start_parallel(gpe::device(mixture), dimGrid, dimBlock, fun);
                                   }
                                   else {
                                       auto fun = [=] __host__ __device__
                                           (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
                                               backward<scalar_t, 3>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx,
                                                                     mixture_a, xes_a,
                                                                     grad_mixture_a, grad_xes_a, grad_output_a,
                                                                     n, requires_grad_mixture, requires_grad_xes);
                                           };
                                       gpe::start_parallel(gpe::device(mixture), dimGrid, dimBlock, fun);
                                   }

                               }));

    return std::make_tuple(grad_mixture, grad_xes);
}

