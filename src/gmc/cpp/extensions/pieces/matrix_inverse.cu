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
namespace matrix_inverse_impl {
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

at::Tensor forward(const torch::Tensor& matrices) {
    using namespace torch::indexing;
    TORCH_CHECK(matrices.sizes().size() >= 2)
    TORCH_CHECK((matrices.size(-1) == 2 || matrices.size(-1) == 3) && matrices.size(-2) == matrices.size(-1))

    const auto original_shape = matrices.sizes().vec();
    const auto n_dims = original_shape.back();
    const auto flattened_matrices = matrices.view({-1, n_dims, n_dims});
    const auto n_batch = uint(flattened_matrices.size(0));

    torch::Tensor inversed_matrices = torch::zeros_like(flattened_matrices);

    const dim3 dimBlock = dim3(128);
    const dim3 dimGrid = dim3((n_batch + dimBlock.x - 1) / dimBlock.x);
    //    std::cout << "forward: dimBlock=" << dimBlock.x << "/" << dimBlock.y << "/" << dimBlock.z << ", dimGrid=" << dimGrid.x << "/" << dimGrid.y << "/" << dimGrid.z << std::endl;


    GPE_DISPATCH_FLOATING_TYPES_AND_DIM(matrices.scalar_type(), n_dims, ([&] {
        const auto matrices_a = gpe::accessor<scalar_t, 3>(flattened_matrices);
        const auto inversed_matrices_a = gpe::accessor<scalar_t, 3>(inversed_matrices);

           auto fun = [matrices_a, inversed_matrices_a, n_batch] __host__ __device__
                (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
                    forward_kernel<scalar_t, N_DIMS>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx, matrices_a, inversed_matrices_a, n_batch);
                };
           gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(matrices), dimGrid, dimBlock, fun);
       }));

    return inversed_matrices.view(original_shape);
}



} // namespace matrix_inverse_impl
} // namespace pieces
