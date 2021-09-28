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

namespace {
// from https://github.com/pytorch/pytorch/blob/0aef44cb3d8a346099ab8cec40258d263d3a262a/torch/csrc/autograd/FunctionsManual.cpp
torch::Tensor eigh_backward(const std::vector<torch::autograd::Variable> &grads, const torch::Tensor& self,
                     bool eigenvectors, const torch::Tensor& L, const torch::Tensor& V) {
    // This function is used for both torch.symeig and torch.linalg.eigh.
    // eigh (and torch.symeig) operates only on symmetric (resp. Hermitian) inputs.

    // General considerations of the differential and adjoint
    // Let U(n) = {U \in C^{n x n} | U^H U = I} by the unitary group and
    // Her(n) = {A \in C^{n x n} | A^H = A} be the Hermitian matrices
    // eigh : Her(n) -> U(n) x R^n
    // Denoting the tangent spaces as T, the differential of eigh at A = VLV^H
    // (i.e. forward differentiation) is a linear map
    // (d eigh)_A : T_A Her(n) -> T_V U(n) x T_L R^n
    // R^n is a linear space, so it is canonically isomorphic to its tangent space
    // Since X, Y \in Her(n) => X + Y \in Her(n), Her(n) is also linear. For this reason, we can write
    // (d eigh)_A : Her(n) -> T_V U(n) x R^n
    // Differentiating the equation U^H U = I, the tangent space of U(n) is given by
    // T_V U(n) = {X \in C^{n x n} | X^H V = -V^H X}. That is, matrices such that V^HX is skew-Hermitian.
    // We then have that the adjoint of the differential (i.e. reverse differentiation) is a map
    // (d eigh)*_A : T_V U(n) x Her(n) -> Her(n)
    // Since the adjoint is defined on T_V U(n), we need to project the input gradient onto T_V U(n)

    // Orthogonal projection \pi_V : C^{n x n} -> T_V U(n)
    // We have that an element gV \in T_V U(n) can be represented as gV = VX for a skew-Hermitian
    // matrix X := V^H gV.
    // Using that V \in U(n) is an isometry of C^{n x n}, we have that
    // \pi_V(gV) := \pi_V(VX) = V\pi_I(X) = V\pi_I(V^H gV)
    // pi_I (X) = (X - X^H) / 2 is the orthogonal projection from C^{n x n} into the skew-Hermitian matrices

    // The formula
    // Following the derivation in
    // https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf (Sec 3.1)
    // For A = VLV^H, with V with unitary and L real,
    // denoting the gradients gA \in Her(n), gV \in C^{n x n} and gL \in R^n, we have
    // gA = (d eigh)*_A(\pi_V(gV), gL)
    //    = V(diag_embed(gL) + \pi_I(V^H gV) / E)V^H
    // where:
    //   - E_ij = L_i - L_j if i != j
    //   - diag_embed takes a vector into a diagonal matrix
    //   - The division by E is done just outside of the diagonal. In the diagonal it is set to zero

    // This check just can be triggered in the backwards of torch.symeig
    TORCH_CHECK(eigenvectors,
                "eigh_backward: torch.symeig(A, eigenvectors=False) is not differentiable. ",
                "Use torch.linalg.eigvalsh(A) instead.");

    const auto gL = grads[0];
    const auto gV = grads[1];

    const auto Vh = V.conj().transpose(-2, -1);

    if (gV.defined()) {
        auto E = L.unsqueeze(-2) - L.unsqueeze(-1);
        if (at::GradMode::is_enabled()) {
            // Avoids differentiating through at infinity when doing gradgrad
            // 1 could be any number, as we are going to overwrite the diagonal
            E.diagonal(0, -2, -1).fill_(1);
        }

        torch::Tensor result =  at::matmul(Vh, gV);
        // Project
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
        result = result.sub(result.transpose(-2, -1).conj()).mul_(0.5);
        // E is skew-symmetric. Multiplying entrywise a skew-Hermitian matrix by a
        // skew-symmetric matrix gives a Hermitian matrix, as we expected.
        result.div_(E);

        if (gL.defined()) {
            result.diagonal(0, -2, -1).copy_(gL);
        }
        else {
            result.diagonal(0, -2, -1).zero_();
        }

        // Conjugating a Hermitian matrix by a unitary matrix gives a Hermitian matrix
        return at::matmul(V, at::matmul(result, Vh));
    }
    else {
        if (gL.defined()) {
            // If we just gL is defined, one matmul suffices
            return at::matmul(V * gL.unsqueeze(-2), Vh);
        } else {
            // If neither is defined, there's nothing to do
            return at::zeros_like(self, at::MemoryFormat::Contiguous);
        }
    }
}
}

std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& matrices) {
    using namespace torch::indexing;
    TORCH_CHECK(matrices.sizes().size() >= 2)
    TORCH_CHECK((matrices.size(-1) == 2 || matrices.size(-1) == 3) && matrices.size(-2) == matrices.size(-1))

    const auto original_shape = matrices.sizes().vec();
    const auto n_dims = original_shape.back();
    const auto flattened_matrices = matrices.view({-1, n_dims * n_dims});
    const auto n_batch = uint(flattened_matrices.size(0));

    torch::Tensor eigenvectors = torch::empty({n_batch, n_dims * n_dims}, torch::device(matrices.device()).dtype(matrices.dtype()));
    torch::Tensor eigenvalues = torch::empty({n_batch, n_dims}, torch::device(matrices.device()).dtype(matrices.dtype()));

    const dim3 dimBlock = dim3(128);
    const dim3 dimGrid = dim3((n_batch + dimBlock.x - 1) / dimBlock.x);


    GPE_DISPATCH_FLOATING_TYPES_AND_DIM(matrices.scalar_type(), n_dims, ([&] {
                                            const auto matrices_a = gpe::struct_accessor<glm::mat<N_DIMS, N_DIMS, scalar_t>, 1>(flattened_matrices);
                                            auto eigenvectors_a = gpe::struct_accessor<glm::mat<N_DIMS, N_DIMS, scalar_t>, 1>(eigenvectors);
                                            auto eigenvalues_a = gpe::struct_accessor<glm::vec<N_DIMS, scalar_t>, 1>(eigenvalues);

                                            auto fun = [matrices_a, eigenvectors_a, eigenvalues_a, n_batch] __host__ __device__ (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) mutable {
                                                using Vec = glm::vec<N_DIMS, scalar_t>;
                                                using Mat = glm::mat<N_DIMS, N_DIMS, scalar_t>;
                                                GPE_UNUSED(gpe_gridDim);
                                                const auto index = gpe_blockDim.x * gpe_blockIdx.x + gpe_threadIdx.x;
                                                if (index >= n_batch)
                                                    return;
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

torch::Tensor backward(const torch::Tensor& matrices, const torch::Tensor& cached_values, const torch::Tensor& cached_vectors, const torch::Tensor& grad_values, const torch::Tensor& grad_vectors) {
    return eigh_backward({grad_values, grad_vectors}, matrices, true, cached_values, cached_vectors);
}


} // namespace matrix_inverse_impl
} // namespace pieces
