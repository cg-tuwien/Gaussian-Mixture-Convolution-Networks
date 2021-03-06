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
#include "parallel_start.h"
#include "util/scalar.h"
#include "util/gaussian.h"
#include "util/gaussian_mixture.h"
#include "util/mixture.h"

namespace {

template <typename scalar_t, int DIMS>
__device__
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

at::Tensor parallel_forward_optimised_impl(const torch::Tensor& mixture, const torch::Tensor& xes) {
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
                                       auto fun = [mixture_a, xes_a, sum_a, n] __device__
                                           (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
                                               forward<scalar_t, 2>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx, mixture_a, xes_a, sum_a, n);
                                           };
                                       gpe::start_parallel<gpe::ComputeDevice::CUDA>(gpe::device(mixture), dimGrid, dimBlock, fun);
                                   }
                                   else {
                                       auto fun = [mixture_a, xes_a, sum_a, n] __device__
                                           (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
                                               forward<scalar_t, 3>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx, mixture_a, xes_a, sum_a, n);
                                           };
                                       gpe::start_parallel<gpe::ComputeDevice::CUDA>(gpe::device(mixture), dimGrid, dimBlock, fun);
                                   }

                                   cudaDeviceSynchronize();
                               }));
    return sum;
}

