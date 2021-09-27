#include "pieces/matrix_inverse.h"

//#include <torch/extension.h>
#include <torch/script.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <algorithm>

#include "common.h"
#include "hacked_accessor.h"
#include "util/cuda.h"
#include "util/glm.h"
#include "parallel_start.h"
#include "math/symeig_detail.h"


namespace pieces {
namespace symeig_impl {

std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& matrices) {
    using namespace torch::indexing;
    TORCH_CHECK(matrices.sizes().size() >= 2)
    TORCH_CHECK((matrices.size(-1) == 2 || matrices.size(-1) == 3) && matrices.size(-2) == matrices.size(-1))

    const auto original_shape = matrices.sizes().vec();
    const auto n_dims = original_shape.back();
    const auto flattened_matrices = matrices.view({-1, n_dims, n_dims});
    const auto n_batch = uint(flattened_matrices.size(0));

    torch::Tensor eigenvectors = torch::empty({n_batch, n_dims, n_dims}, torch::device(matrices.device()).dtype(matrices.dtype()));
    torch::Tensor eigenvalues = torch::empty({n_batch, n_dims}, torch::device(matrices.device()).dtype(matrices.dtype()));

    const dim3 dimBlock = dim3(128);
    const dim3 dimGrid = dim3((n_batch + dimBlock.x - 1) / dimBlock.x);


    GPE_DISPATCH_FLOATING_TYPES_AND_DIM(matrices.scalar_type(), n_dims, ([&] {
        const auto matrices_a = gpe::struct_accessor<glm::mat<N_DIMS, N_DIMS, scalar_t>, 1>(flattened_matrices);
        auto eigenvectors_a = gpe::struct_accessor<glm::mat<N_DIMS, N_DIMS, scalar_t>, 1>(eigenvectors);
        auto eigenvalues_a = gpe::struct_accessor<glm::vec<N_DIMS, scalar_t>, 1>(eigenvalues);

        auto fun = [matrices_a, eigenvectors_a, eigenvalues_a] __host__ __device__ (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) mutable {
            using Vec = glm::vec<N_DIMS, scalar_t>;
            using Mat = glm::mat<N_DIMS, N_DIMS, scalar_t>;
            GPE_UNUSED(gpe_gridDim);
            const auto index = gpe_blockDim.x * gpe_blockIdx.x + gpe_threadIdx.x;
            const auto& matrix = matrices_a[index];
            Mat& eigenvectors = eigenvectors_a[index];
            Vec& eigenvalues = eigenvalues_a[index];
            thrust::tie(eigenvalues, eigenvectors) = gpe::detail::compute_symeig(matrix);

        };
       gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(matrices), dimGrid, dimBlock, fun);
       }));

    auto eigenvalue_sizes = original_shape;
    eigenvalue_sizes.pop_back();

    return {eigenvalues.view(eigenvalue_sizes), eigenvectors.view(original_shape)};
}



} // namespace matrix_inverse_impl
} // namespace pieces
