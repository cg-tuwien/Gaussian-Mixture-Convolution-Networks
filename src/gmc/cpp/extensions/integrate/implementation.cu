#include "integrate/implementation.h"

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
#include "util/mixture.h"
#include "util/gaussian_mixture.h"

namespace integrate {
namespace {

template <typename scalar_t, int N_DIMS, bool INVERSED>
__host__ __device__
void forward(const dim3& gpe_gridDim, const dim3& gpe_blockDim,
             const dim3& gpe_blockIdx, const dim3& gpe_threadIdx,
             const gpe::PackedTensorAccessor32<scalar_t, 4> mixture_a,
             gpe::PackedTensorAccessor32<scalar_t, 3> integrands_a,
             const gpe::MixtureNs n) {
    GPE_UNUSED(gpe_gridDim)
    const auto batch_index = int(gpe_blockIdx.z);
    const auto layer_index = int(gpe_blockIdx.y);
    const auto component_index = int(gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x);

    if (component_index >= n.components)
        return;
    
    const auto& g = gpe::gaussian<N_DIMS>(mixture_a[batch_index][layer_index][component_index]);
    if (INVERSED)
        integrands_a[batch_index][layer_index][component_index] = gpe::integrate_inversed(g);
    else
        integrands_a[batch_index][layer_index][component_index] = gpe::integrate(g);
}


/*
template <typename scalar_t, int DIMS>
__host__ __device__
void backward(const dim3& gpe_gridDim, const dim3& gpe_blockDim,
              const dim3& gpe_blockIdx, const dim3& gpe_threadIdx,
              const gpe::PackedTensorAccessor32<scalar_t, 4> mixture_a,
              const gpe::PackedTensorAccessor32<scalar_t, 4> xes_a,
              gpe::PackedTensorAccessor32<scalar_t, 4> grad_mixture_a,
              gpe::PackedTensorAccessor32<scalar_t, 4> grad_xes_a,
              const gpe::PackedTensorAccessor32<scalar_t, 3> grad_output_a,
              const gpe::MixtureAndXesNs n, bool requires_grad_mixture, bool requires_grad_xes) {
    GPE_UNUSED(gpe_gridDim)
    const auto batch_index = int(gpe_blockIdx.z);
    const auto layer_index = int(gpe_blockIdx.y);
    const auto batch_xes_index = gpe::min(batch_index, n.batch_xes - 1);
    const auto layer_xes_index = gpe::min(layer_index, n.layers_xes - 1);
    const auto xes_index = int(gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x);

    if (xes_index >= n.xes)
        return;
       printf("block %d/%d/%d, thread %d: batch_index=%d, layer_index=%d, component_index=%d, xes_index=%d \n",
              blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x,
              batch_index, layer_index, component_index, xes_index);

    for (int component_index = 0; component_index < n.components; ++component_index) {
        auto a = xes_a[batch_xes_index][layer_xes_index][xes_index];
        const scalar_t& memory_location = a[0];
        const auto& x_pos = gpe::vec<DIMS>(memory_location);

           glm::vec<DIMS, scalar_t>& grad_xes = gpe::vec<DIMS>(grad_xes_a[batch_layer_index][xes_index][0]);

           auto& grad_c_weight = gpe::weight(grad_mixture_a[batch_layer_index][component_index]);
           auto& grad_c_pos = gpe::position<DIMS>(grad_mixture_a[batch_layer_index][component_index]);
           auto& grad_c_cov = gpe::covariance<DIMS>(grad_mixture_a[batch_layer_index][component_index]);

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
            for (int i = 0; i < DIMS; ++i) {
                gpe::atomicAdd(&grad_xes_a[batch_xes_index][layer_xes_index][xes_index][i], grad_xes_addition[i]);
            }
        }
        if (requires_grad_mixture) {
            const auto grad_c_weight_addition = exp * grad_output_a[batch_index][layer_index][xes_index];
            const auto grad_c_pos_addition = local_grad_c_pos * grad_output_a[batch_index][layer_index][xes_index];
            const auto grad_c_cov_addition = - c_weight * scalar_t(0.5) * exp * grad_output_a[batch_index][layer_index][xes_index] * glm::outerProduct(t, t);
            gpe::atomicAdd(&grad_mixture_a[batch_index][layer_index][component_index][0], grad_c_weight_addition);
            for (int i = 0; i < DIMS; ++i) {
                gpe::atomicAdd(&grad_mixture_a[batch_index][layer_index][component_index][1 + i], grad_c_pos_addition[i]);
                for (int j = 0; j < DIMS; ++j)
                    gpe::atomicAdd(&grad_mixture_a[batch_index][layer_index][component_index][1 + DIMS + i*DIMS + j], grad_c_cov_addition[i][j]);
            }
        }
    }
}
*/

} // anonymous namespace

template <bool INVERSED>
at::Tensor forward_impl(const torch::Tensor& mixture) {
    using namespace torch::indexing;
    auto n = gpe::get_ns(mixture);

    torch::Tensor integrands = torch::zeros({n.batch, n.layers, n.components}, torch::dtype(mixture.dtype()).device(mixture.device()));

    TORCH_CHECK(n.batch * n.layers < 65535, "n_batch x n_layers must be smaller than 65535 for CUDA")


    dim3 dimBlock = dim3(128, 1, 1);
    const dim3 dimGrid = dim3((uint(n.components) + dimBlock.x - 1) / dimBlock.x,
                              uint(n.layers),
                              uint(n.batch));
    //    std::cout << "forward: dimBlock=" << dimBlock.x << "/" << dimBlock.y << "/" << dimBlock.z << ", dimGrid=" << dimGrid.x << "/" << dimGrid.y << "/" << dimGrid.z << std::endl;


    GPE_DISPATCH_FLOATING_TYPES_AND_DIM(mixture.scalar_type(), n.dims, ([&] {
                                   auto integrands_a = gpe::accessor<scalar_t, 3>(integrands);
                                   auto mixture_a = gpe::accessor<scalar_t, 4>(mixture);

                                   auto fun = [mixture_a, integrands_a, n] __host__ __device__
                                        (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
                                            forward<scalar_t, N_DIMS, INVERSED>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx, mixture_a, integrands_a, n);
                                        };
                                   gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(mixture), dimGrid, dimBlock, fun);
                               }));
    
    return integrands.sum(-1);
}

template at::Tensor forward_impl<true>(const torch::Tensor& mixture);
template at::Tensor forward_impl<false>(const torch::Tensor& mixture);

/*
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

    dim3 dimBlock = dim3(128, 1, 1);
    const dim3 dimGrid = dim3((uint(n.xes) + dimBlock.x - 1) / dimBlock.x,
                              uint(n.layers),
                              uint(n.batch));
    //    std::cout << "forward: dimBlock=" << dimBlock.x << "/" << dimBlock.y << "/" << dimBlock.z << ", dimGrid=" << dimGrid.x << "/" << dimGrid.y << "/" << dimGrid.z << std::endl;

    AT_DISPATCH_FLOATING_TYPES(mixture.scalar_type(), "cuda_parallel_backward_impl", ([&] {
                                   auto mixture_a = gpe::accessor<scalar_t, 4>(mixture);
                                   auto xes_a = gpe::accessor<scalar_t, 4>(xes);
                                   auto grad_mixture_a = gpe::accessor<scalar_t, 4>(grad_mixture);
                                   auto grad_xes_a = gpe::accessor<scalar_t, 4>(grad_xes);
                                   auto grad_output_a = gpe::accessor<scalar_t, 3>(grad_output);

                                   if (n.dims == 2) {
                                       auto fun = [=] __host__ __device__
                                           (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
                                               backward<scalar_t, 2>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx,
                                                                     mixture_a, xes_a,
                                                                     grad_mixture_a, grad_xes_a, grad_output_a,
                                                                     n, requires_grad_mixture, requires_grad_xes);
                                           };
                                       gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(mixture), dimGrid, dimBlock, fun);
                                   }
                                   else {
                                       auto fun = [=] __host__ __device__
                                           (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
                                               backward<scalar_t, 3>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx,
                                                                     mixture_a, xes_a,
                                                                     grad_mixture_a, grad_xes_a, grad_output_a,
                                                                     n, requires_grad_mixture, requires_grad_xes);
                                           };
                                       gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(mixture), dimGrid, dimBlock, fun);
                                   }

                               }));

    return std::make_tuple(grad_mixture, grad_xes);
}
*/

} // namespace integrate
