#include "evaluate_inversed/implementations.h"

//#include <torch/extension.h>
#include <torch/script.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <algorithm>

#include "cuda_qt_creator_definitinos.h"
#include "common.h"
#include "hacked_accessor.h"
#include "parallel_start.h"
#include "util/scalar.h"
#include "util/gaussian.h"
#include "util/gaussian_mixture.h"
#include "util/grad/gaussian.h"
#include "util/mixture.h"

namespace {

template <typename scalar_t>
__device__
void reduce_warp(scalar_t v, scalar_t* dest)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    if(!(threadIdx.x & 0x1F)) // warp leader
    {
        atomicAdd(dest, v);
    }
}

template <typename scalar_t, int DIMS>
__device__
void backward(const dim3& gpe_gridDim, const dim3& gpe_blockDim,
              const dim3& gpe_blockIdx, const dim3& gpe_threadIdx,
              const gpe::PackedTensorAccessor32<scalar_t, 4> mixture_a,
              const gpe::PackedTensorAccessor32<scalar_t, 4> xes_a,
              gpe::PackedTensorAccessor32<scalar_t, 4> grad_mixture_a,
              gpe::PackedTensorAccessor32<scalar_t, 4> grad_xes_a,
              const gpe::PackedTensorAccessor32<scalar_t, 3> grad_output_a,
              const gpe::MixtureAndXesNs n, bool requires_grad_mixture, bool requires_grad_xes) {
    using Gaussian = gpe::Gaussian<DIMS, scalar_t>;
    using Vec = typename Gaussian::pos_t;
    GPE_UNUSED(gpe_gridDim)
    const auto batch_index = int(gpe_blockIdx.z);
    const auto layer_index = int(gpe_blockIdx.y);
    const auto batch_xes_index = gpe::min(batch_index, n.batch_xes - 1);
    const auto layer_xes_index = gpe::min(layer_index, n.layers_xes - 1);
    const auto xes_index = int(gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x);


    if (xes_index >= n.xes)
        return;
    //    printf("block %d/%d/%d, thread %d: batch_index=%d, layer_index=%d, component_index=%d, xes_index=%d \n",
    //           blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x,
    //           batch_index, layer_index, component_index, xes_index);

    for (int component_index = 0; component_index < n.components; ++component_index) { // component index atm up to 1280, in the future may be 10-20k
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

        Gaussian grad_component = {};
        Vec grad_x_pos = {};
        gpe::grad::evaluate_inversed(Gaussian(c_weight, c_pos, c_cov), x_pos, &grad_component, &grad_x_pos, grad_output_a[batch_index][layer_index][xes_index]);

        if (requires_grad_xes) {
            for (int i = 0; i < DIMS; ++i) {
                gpe::atomicAdd(&grad_xes_a[batch_xes_index][layer_xes_index][xes_index][i], grad_x_pos[i]);
            }
        }


        if (requires_grad_mixture) {
            //gpe::atomicAdd(&grad_mixture_a[batch_index][layer_index][component_index][0], grad_c_weight_addition);
            reduce_warp(grad_component.weight, &grad_mixture_a[batch_index][layer_index][component_index][0]);
            for (int i = 0; i < DIMS; ++i) {
                //gpe::atomicAdd(&grad_mixture_a[batch_index][layer_index][component_index][1 + i], grad_c_pos_addition[i]);
                reduce_warp(grad_component.position[i], &grad_mixture_a[batch_index][layer_index][component_index][1 + i]);
                //atomicAdd(&temp[1 + i], grad_c_pos_addition[i]);
                for (int j = 0; j < DIMS; ++j)
                {
                    reduce_warp(grad_component.covariance[i][j], &grad_mixture_a[batch_index][layer_index][component_index][1 + DIMS + i*DIMS + j]);
                    //gpe::atomicAdd(&grad_mixture_a[batch_index][layer_index][component_index][1 + DIMS + i*DIMS + j], grad_c_cov_addition[i][j]);
                    //atomicAdd(&temp[1 + DIMS + i*DIMS + j], grad_c_cov_addition[i][j]);
                }
            }
        }
    }
}

}

std::tuple<torch::Tensor, torch::Tensor> parallel_backward_optimised_impl(const torch::Tensor& grad_output, const torch::Tensor& mixture, const torch::Tensor& xes, bool requires_grad_mixture, bool requires_grad_xes) {
    gpe::check_mixture(mixture);
    auto n = gpe::check_input_and_get_ns(mixture, xes);

    TORCH_CHECK(grad_output.dim() == 3, "grad_output has wrong number of dimensions")
    TORCH_CHECK(grad_output.size(0) == n.batch, "grad_output has wrong batch dimension")
    TORCH_CHECK(grad_output.size(1) == n.layers, "grad_output has wrong layer dimension")
    TORCH_CHECK(grad_output.size(2) == n.xes, "grad_output has wrong xes dimension")
    TORCH_CHECK(grad_output.dtype() == mixture.dtype(), "grad_output dtype does not match with mixture dtype")

    // mixture[n.batch(10-100)][n.layers(10-64)][n.components(2000-20000)][daten(7, 13)]
    // xes[1|n.batch][1|n.layers][n.xes(100x100|n.components][n.dims(2, 3)]
    // grad_output[n.batch][n.layers][n.xes]
    // grad_mixture same as mixture
    // grad_xes same as xes

    torch::Tensor mixture_view = mixture.view({-1, n.components, mixture.size(3)});

    torch::Tensor grad_mixture = torch::zeros({n.batch, n.layers, n.components, mixture.size(3)}, torch::dtype(mixture.dtype()).device(mixture.device()));
    torch::Tensor grad_xes = torch::zeros({n.batch_xes, n.layers_xes, n.xes, n.dims}, torch::dtype(mixture.dtype()).device(mixture.device()));

    dim3 dimBlock = dim3(128, 1, 1);
    const dim3 dimGrid = dim3((uint(n.xes) + dimBlock.x - 1) / dimBlock.x,
                              uint(n.layers),
                              uint(n.batch));
    //    std::cout << "forward: dimBlock=" << dimBlock.x << "/" << dimBlock.y << "/" << dimBlock.z << ", dimGrid=" << dimGrid.x << "/" << dimGrid.y << "/" << dimGrid.z << std::endl;

    AT_DISPATCH_FLOATING_TYPES(mixture.scalar_type(), "cuda_parallel_backward_impl", ([&] {
                                   const auto mixture_a = gpe::accessor<scalar_t, 4>(mixture);
                                   const auto xes_a = gpe::accessor<scalar_t, 4>(xes);
                                   auto grad_mixture_a = gpe::accessor<scalar_t, 4>(grad_mixture);
                                   auto grad_xes_a = gpe::accessor<scalar_t, 4>(grad_xes);
                                   const auto grad_output_a = gpe::accessor<scalar_t, 3>(grad_output);

                                   if (n.dims == 2) {
                                       auto fun = [=] __device__
                                           (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
                                               backward<scalar_t, 2>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx,
                                                                     mixture_a, xes_a,
                                                                     grad_mixture_a, grad_xes_a, grad_output_a,
                                                                     n, requires_grad_mixture, requires_grad_xes);
                                           };
                                       gpe::start_parallel<gpe::ComputeDevice::CUDA>(gpe::device(mixture), dimGrid, dimBlock, fun);
                                   }
                                   else {
                                       auto fun = [=] __device__
                                           (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
                                               backward<scalar_t, 3>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx,
                                                                     mixture_a, xes_a,
                                                                     grad_mixture_a, grad_xes_a, grad_output_a,
                                                                     n, requires_grad_mixture, requires_grad_xes);
                                           };
                                       gpe::start_parallel<gpe::ComputeDevice::CUDA>(gpe::device(mixture), dimGrid, dimBlock, fun);
                                   }

                               }));

    return std::make_tuple(grad_mixture, grad_xes);
}

