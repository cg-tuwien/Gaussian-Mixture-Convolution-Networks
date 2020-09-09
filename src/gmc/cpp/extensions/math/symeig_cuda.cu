////#include <torch/extension.h>
#include <torch/script.h>

//#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <vector>
#include <tuple>
#include <algorithm>
#include <thrust/tuple.h>

#include "math/matrix.h"
#include "math/symeig_detail.h"
#include "common.h"

#ifndef __CUDACC__
constexpr dim3 blockIdx;
constexpr dim3 blockDim;
constexpr dim3 threadIdx;
using std::min;
using std::max;
#endif

template <typename scalar_t, int DIMS>
__global__ void kernel_forward(const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> matrices_a,
                               torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> eigenvalues_a,
                               torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> eigenvectors_a,
                               const uint n_batch) {
    using Vec = glm::vec<DIMS, scalar_t>;
    using Mat = glm::mat<DIMS, DIMS, scalar_t>;

    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_batch)
        return;

    const auto& mat = gpe::mat<DIMS>(matrices_a[i][0][0]);
    Vec& eigenvalues = gpe::vec<DIMS>(eigenvalues_a[i][0]);
    Mat& eigenvectors = gpe::mat<DIMS>(eigenvectors_a[i][0][0]);
    thrust::tie(eigenvalues, eigenvectors) = gpe::detail::compute_symeig(mat);
}


std::vector<torch::Tensor> symeig_cuda_forward_impl(const torch::Tensor& matrices) {
    using namespace torch::indexing;
    // currently only 2x2 matrices
    TORCH_CHECK(matrices.sizes().size() >= 2);
    TORCH_CHECK(matrices.size(-1) == 2 && matrices.size(-2) == 2);
    TORCH_CHECK(matrices.device().is_cuda(), "this one is just for cuda..");

    const auto original_shape = matrices.sizes().vec();
    const auto n_dims = original_shape.back();
    const auto flattened_matrices = matrices.view({-1, n_dims, n_dims});
    const uint n_batch = flattened_matrices.size(0);

    torch::Tensor eigenvectors = torch::zeros_like(flattened_matrices);
    torch::Tensor eigenvalues = torch::zeros({n_batch, n_dims}, at::TensorOptions(matrices.device()).dtype(matrices.dtype()));

    const dim3 dimBlock = dim3(128);
    const dim3 dimGrid = dim3((n_batch + dimBlock.x - 1) / dimBlock.x);
    std::cout << "dimBlock=" << dimBlock.x << "/" << dimBlock.y << "/" << dimBlock.z << "  dimGrid=" << dimGrid.x << "/" << dimGrid.y << "/" << dimGrid.z << std::endl;

    AT_DISPATCH_FLOATING_TYPES(matrices.scalar_type(), "eval_inversed_omp", ([&] {
        const auto matrices_a = flattened_matrices.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto eigenvalues_a = eigenvalues.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>();
        auto eigenvectors_a = eigenvectors.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();

        if (n_dims == 2)
            kernel_forward<scalar_t, 2><<<dimGrid, dimBlock>>>(matrices_a, eigenvalues_a, eigenvectors_a, n_batch);
//        else
//            execute_parallel_forward<scalar_t, 3>(mixture_a, xes_a, sum_a, n);
        cudaDeviceSynchronize();
    }));
    cudaDeviceSynchronize();
    std::cout << "done" << std::endl;
    auto eigenvalues_shape = original_shape;

    eigenvalues_shape.pop_back();

    return {eigenvalues.view(eigenvalues_shape), eigenvectors.view(original_shape)};
}

