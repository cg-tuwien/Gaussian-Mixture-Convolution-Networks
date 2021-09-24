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
#include "util/mixture.h"
#include "parallel_start.h"


namespace pieces {
namespace symeig_impl {
namespace {
template <typename scalar_t, int DIMS> EXECUTION_DEVICES
void forward_kernel(const dim3& gpe_gridDim, const dim3& gpe_blockDim,
             const dim3& gpe_blockIdx, const dim3& gpe_threadIdx,
             const gpe::PackedTensorAccessor32<scalar_t, 3, gpe::RestrictPtrTraits> matrices,
             gpe::PackedTensorAccessor32<scalar_t, 3, gpe::RestrictPtrTraits> inversed_matrices,
             const uint n_batch) {
    GPE_UNUSED(gpe_gridDim)

    using Mat = glm::mat<DIMS, DIMS, scalar_t>;

    const auto i = int(gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x);
    if (i >= int(n_batch))
        return;

    const auto& mat = gpe::mat<DIMS>(matrices[i][0][0]);
    Mat& inv_mat = gpe::mat<DIMS>(inversed_matrices[i][0][0]);
    inv_mat = glm::inverse(mat);
}

} // anonymous namespace

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
        const auto eigenvectors_a = gpe::struct_accessor<glm::mat<N_DIMS, N_DIMS, scalar_t>, 1>(eigenvectors);
        const auto eigenvalues_a = gpe::struct_accessor<glm::vec<N_DIMS, scalar_t>, 1>(eigenvalues);

           auto fun = [matrices_a, eigenvectors_a, eigenvalues_a] __host__ __device__
                (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
                    implement forward computaion, see symeig_detail.h
                };
           gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(matrices), dimGrid, dimBlock, fun);
       }));

    return views on eigenvectors and eigenvalues
}



} // namespace matrix_inverse_impl
} // namespace pieces
