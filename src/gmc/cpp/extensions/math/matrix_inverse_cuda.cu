////#include <torch/extension.h>
#include <torch/script.h>

//#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <vector>
#include <tuple>
#include <algorithm>
#include <thrust/tuple.h>

#include "common.h"
#include "cuda_qt_creator_definitinos.h"
#include "math/symeig_detail.h"

template <typename scalar_t, int DIMS>
__global__ void kernel_forward(const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> matrices,
                               torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> inversed_matrices,
                               const uint n_batch) {
    using Mat = glm::mat<DIMS, DIMS, scalar_t>;

    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_batch)
        return;

    const auto& mat = gpe::mat<DIMS>(matrices[i][0][0]);
    Mat& inv_mat = gpe::mat<DIMS>(inversed_matrices[i][0][0]);
    inv_mat = glm::inverse(mat);
}


torch::Tensor matrix_inverse_cuda_forward_impl(const torch::Tensor& matrices) {
    using namespace torch::indexing;
    TORCH_CHECK(matrices.sizes().size() >= 2);
    TORCH_CHECK((matrices.size(-1) == 2 || matrices.size(-1) == 3) && matrices.size(-2) == matrices.size(-1));
    TORCH_CHECK(matrices.device().is_cuda(), "this one is just for cuda..");

    const auto original_shape = matrices.sizes().vec();
    const auto n_dims = original_shape.back();
    const auto flattened_matrices = matrices.view({-1, n_dims, n_dims});
    const uint n_batch = flattened_matrices.size(0);

    torch::Tensor inversed_matrices = torch::zeros_like(flattened_matrices);

    const dim3 dimBlock = dim3(128);
    const dim3 dimGrid = dim3((n_batch + dimBlock.x - 1) / dimBlock.x);
//    std::cout << "dimBlock=" << dimBlock.x << "/" << dimBlock.y << "/" << dimBlock.z << "  dimGrid=" << dimGrid.x << "/" << dimGrid.y << "/" << dimGrid.z << std::endl;

    AT_DISPATCH_FLOATING_TYPES(matrices.scalar_type(), "matrix_inverse_cuda", ([&] {
        const auto matrices_a = flattened_matrices.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        const auto inversed_matrices_a = inversed_matrices.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();

        if (n_dims == 2)
            kernel_forward<scalar_t, 2><<<dimGrid, dimBlock>>>(matrices_a, inversed_matrices_a, n_batch);
        else
            kernel_forward<scalar_t, 3><<<dimGrid, dimBlock>>>(matrices_a, inversed_matrices_a, n_batch);
    }));
//    cudaDeviceSynchronize();

    return inversed_matrices.view(original_shape);
}

