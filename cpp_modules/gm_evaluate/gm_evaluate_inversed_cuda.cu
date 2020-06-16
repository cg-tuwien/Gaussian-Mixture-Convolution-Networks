#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <algorithm>

#include "common.h"

#ifndef __CUDACC__
constexpr dim3 blockIdx;
constexpr dim3 blockDim;
constexpr dim3 threadIdx;
#endif

namespace gm {
__forceinline__ __device__ float exp(float x) {
    return ::expf(x);
}
__forceinline__ __device__ double exp(double x) {
    return ::exp(x);
}

}

template <typename scalar_t, int DIMS>
__global__ void kernel_forward(const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> mixture_a,
                      const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> xes_a,
                      torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sum_a,
                      const gm::Ns n) {
    const auto batch_layer_index = blockIdx.z;
    const auto xes_index = blockIdx.y;
    const auto component_index = blockIdx.x * blockDim.x + threadIdx.x;;
    if (component_index >= uint(n.components))
        return;

//    printf("block %d/%d/%d, thread %d: batch_layer_index=%d, xes_index=%d, component_index=%d \n",
//           blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x,
//           batch_layer_index, xes_index, component_index);

    const auto& x_pos = gm::vec<DIMS>(xes_a[batch_layer_index][xes_index][0]);

    const auto& c_weight = gm::weight(mixture_a[batch_layer_index][component_index]);
    const auto& c_pos = gm::position<DIMS>(mixture_a[batch_layer_index][component_index]);
    const auto& c_cov = gm::covariance<DIMS>(mixture_a[batch_layer_index][component_index]);
    const auto t = x_pos - c_pos;
    const auto v = scalar_t(-0.5) * glm::dot(t, (c_cov * t));
    const auto w = c_weight * gm::exp(v);

//    sum_a[batch_layer_index][xes_index] += w; // test performance impact of atomicAdd; but this will give a wrong result.
    atomicAdd(&sum_a[batch_layer_index][xes_index], w);
}


template <typename scalar_t, int DIMS>
__global__ void kernel_backward(const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> mixture_a,
                      const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> xes_a,
                      torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_mixture_a,
                      torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_xes_a,
                      torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_output_a,
                      const gm::Ns n, bool requires_grad_mixture, bool requires_grad_xes) {

    const auto batch_layer_index = blockIdx.z;
    const auto xes_index = blockIdx.y;
    const auto component_index = blockIdx.x * blockDim.x + threadIdx.x;;
    if (component_index >= uint(n.components))
        return;

    const auto& x_pos = gm::vec<DIMS>(xes_a[batch_layer_index][xes_index][0]);

//    glm::vec<DIMS, scalar_t>& grad_xes = gm::vec<DIMS>(grad_xes_a[batch_layer_index][xes_index][0]);

//    auto& grad_c_weight = gm::weight(grad_mixture_a[batch_layer_index][component_index]);
//    auto& grad_c_pos = gm::position<DIMS>(grad_mixture_a[batch_layer_index][component_index]);
//    auto& grad_c_cov = gm::covariance<DIMS>(grad_mixture_a[batch_layer_index][component_index]);

    const auto& c_weight = gm::weight(mixture_a[batch_layer_index][component_index]);
    const auto& c_pos = gm::position<DIMS>(mixture_a[batch_layer_index][component_index]);
    const auto& c_cov = gm::covariance<DIMS>(mixture_a[batch_layer_index][component_index]);

    const auto t = x_pos - c_pos;
    const auto v = scalar_t(-0.5) * glm::dot(t, (c_cov * t));
    const auto exp = gm::exp(v);
    const auto weighted_exp = c_weight * exp;
    const auto local_grad_c_pos = weighted_exp * t * c_cov;

    if (requires_grad_xes) {
        const auto grad_xes_addition = - grad_output_a[batch_layer_index][xes_index] * local_grad_c_pos;
        for (uint i = 0; i < DIMS; ++i) {
            atomicAdd(&grad_xes_a[batch_layer_index][xes_index][i], grad_xes_addition[i]);
        }
    }
    if (requires_grad_mixture) {
        const auto grad_c_weight_addition = exp * grad_output_a[batch_layer_index][xes_index];
        const auto grad_c_pos_addition = local_grad_c_pos * grad_output_a[batch_layer_index][xes_index];
        const auto grad_c_cov_addition = - c_weight * scalar_t(0.5) * exp * grad_output_a[batch_layer_index][xes_index] * glm::outerProduct(t, t);
        atomicAdd(&grad_mixture_a[batch_layer_index][component_index][0], grad_c_weight_addition);
        for (uint i = 0; i < DIMS; ++i) {
            atomicAdd(&grad_mixture_a[batch_layer_index][component_index][1 + i], grad_c_pos_addition[i]);
            for (uint j = 0; j < DIMS; ++j)
                atomicAdd(&grad_mixture_a[batch_layer_index][component_index][1 + DIMS + i*DIMS + j], grad_c_cov_addition[i][j]);
        }
    }

}

torch::Tensor cuda_evaluate_inversed_forward(torch::Tensor mixture, torch::Tensor xes) {
    using namespace torch::indexing;
    auto n = gm::check_input_and_get_ns(mixture, xes);

    mixture = mixture.view({n.batch * n.layers, n.components, -1});
    xes = xes.view({n.batch_xes * n.layers_xes, n.xes, n.dims});
    torch::Tensor sum = torch::zeros({n.batch * n.layers, n.xes}, torch::dtype(mixture.dtype()).device(mixture.device()));

    TORCH_CHECK(mixture.device().is_cuda(), "mixture must be a CUDA tensor");
    TORCH_CHECK(n.batch * n.layers < 65535, "n_batch x n_layers must be smaller than 65535 for CUDA");
    TORCH_CHECK(n.xes < 65535, "number of xes must be smaller than 65535 for CUDA");


    constexpr dim3 dimBlock = dim3(128, 1, 1);
    const dim3 dimGrid = dim3((n.components + dimBlock.x - 1) / dimBlock.x,
                              n.xes,
                              n.batch * n.layers);
//    std::cout << "forward: dimBlock.x=" << dimBlock.x << ", n_blocks=" << dimGrid.x << "/" << dimGrid.y << "/" << dimGrid.z << std::endl;


    AT_DISPATCH_FLOATING_TYPES(mixture.scalar_type(), "eval_inversed_omp", ([&] {
        auto mixture_a = mixture.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto xes_a = xes.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto sum_a = sum.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>();

        if (n.dims == 2)
            kernel_forward<scalar_t, 2><<<dimGrid, dimBlock>>>(mixture_a, xes_a, sum_a, n);
        else
            kernel_forward<scalar_t, 3><<<dimGrid, dimBlock>>>(mixture_a, xes_a, sum_a, n);
    }));
    return sum.view({n.batch, n.layers, n.xes});
}

std::vector<torch::Tensor> cuda_evaluate_inversed_backward(torch::Tensor grad_output, torch::Tensor mixture, torch::Tensor xes, bool requires_grad_mixture, bool requires_grad_xes) {
    gm::check_mixture(mixture);
    auto n = gm::check_input_and_get_ns(mixture, xes);

    TORCH_CHECK(mixture.device().is_cuda(), "mixture must be a CUDA tensor")
    TORCH_CHECK(grad_output.device().is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(grad_output.dim() == 3, "grad_output has wrong number of dimensions");
    TORCH_CHECK(grad_output.size(0) == n.batch, "grad_output has wrong batch dimension");
    TORCH_CHECK(grad_output.size(1) == n.layers, "grad_output has wrong layer dimension");
    TORCH_CHECK(grad_output.size(2) == n.xes, "grad_output has wrong xes dimension");
    TORCH_CHECK(grad_output.dtype() == mixture.dtype(), "grad_output dtype does not match with mixture dtype")


    torch::Tensor grad_mixture = torch::zeros({n.batch * n.layers, n.components, mixture.size(3)}, torch::dtype(mixture.dtype()).device(mixture.device()));
    torch::Tensor grad_xes = torch::zeros({n.batch_xes * n.layers_xes, n.xes, n.dims}, torch::dtype(mixture.dtype()).device(mixture.device()));

    grad_output = grad_output.view({n.batch * n.layers, n.xes});
    mixture = mixture.view({n.batch * n.layers, n.components, -1});
    xes = xes.view({n.batch_xes * n.layers_xes, n.xes, n.dims});

    constexpr dim3 dimBlock = dim3(128, 1, 1);
    const dim3 dimGrid = dim3((n.components + dimBlock.x - 1) / dimBlock.x,
                              n.xes,
                              n.batch * n.layers);
//    std::cout << "backward: dimBlock.x=" << dimBlock.x << ", n_blocks=" << dimGrid.x << "/" << dimGrid.y << "/" << dimGrid.z << std::endl;

    AT_DISPATCH_FLOATING_TYPES(mixture.scalar_type(), "eval_inversed_omp_backward", ([&] {
        auto mixture_a = mixture.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto xes_a = xes.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto grad_mixture_a = grad_mixture.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto grad_xes_a = grad_xes.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto grad_output_a = grad_output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>();

        if (n.dims == 2)
            kernel_backward<scalar_t, 2><<<dimGrid, dimBlock>>>(mixture_a, xes_a, grad_mixture_a, grad_xes_a, grad_output_a, n, requires_grad_mixture, requires_grad_xes);
        else
            kernel_backward<scalar_t, 3><<<dimGrid, dimBlock>>>(mixture_a, xes_a, grad_mixture_a, grad_xes_a, grad_output_a, n, requires_grad_mixture, requires_grad_xes);
    }));

    return {grad_mixture.view({n.batch, n.layers, n.components, -1}), grad_xes.view({n.batch_xes, n.layers_xes, n.xes, n.dims})};
}
