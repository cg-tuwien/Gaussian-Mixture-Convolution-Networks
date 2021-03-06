#include "math/symeig_cpu.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include <omp.h>

#include <thrust/tuple.h>

#include <torch/extension.h>

#include "util/glm.h"

#include "common.h"
#include "math/symeig_detail.h"
#include "util/mixture.h"

namespace {

template <typename scalar_t, int DIMS>
void execute_parallel_forward(const torch::PackedTensorAccessor32<scalar_t, 3>& matrices_a,
                      torch::PackedTensorAccessor32<scalar_t, 3>& eigenvectors_a,
                      torch::PackedTensorAccessor32<scalar_t, 2>& eigenvalues_a,
                      const int& n_batch) {
    using Vec = glm::vec<DIMS, scalar_t>;
    using Mat = glm::mat<DIMS, DIMS, scalar_t>;

    #pragma omp parallel for num_threads(omp_get_num_procs())
    for (int i = 0; i < n_batch; ++i) {
        const auto& mat = gpe::mat<DIMS>(matrices_a[i][0][0]);
        Vec& eigenvalues = gpe::vec<DIMS>(eigenvalues_a[i][0]);
        Mat& eigenvectors = gpe::mat<DIMS>(eigenvectors_a[i][0][0]);
        thrust::tie(eigenvalues, eigenvectors) = gpe::detail::compute_symeig(mat);
    }
}
}

std::tuple<at::Tensor, at::Tensor> gpe::symeig_cpu_forward(const torch::Tensor& matrices) {
    using namespace torch::indexing;
    // currently only 2x2 matrices
    TORCH_CHECK((matrices.size(-1) == 2 && matrices.size(-2) == 2) || (matrices.size(-1) == 3 && matrices.size(-2) == 3))
    TORCH_CHECK(matrices.device().is_cpu(), "this one is just for cpu..")

    const auto original_shape = matrices.sizes().vec();
    const auto n_dims = original_shape.back();
    const auto flattened_matrices = matrices.view({-1, n_dims, n_dims});
    const int n_batch = int(flattened_matrices.size(0));

    torch::Tensor eigenvectors = torch::zeros_like(flattened_matrices);
    torch::Tensor eigenvalues = torch::zeros({n_batch, n_dims}, at::TensorOptions(matrices.device()).dtype(matrices.dtype()));

    AT_DISPATCH_FLOATING_TYPES(matrices.scalar_type(), "eval_inversed_omp", ([&] {
        auto matrices_a = flattened_matrices.packed_accessor32<scalar_t, 3>();
        auto eigenvectors_a = eigenvectors.packed_accessor32<scalar_t, 3>();
        auto eigenvalues_a = eigenvalues.packed_accessor32<scalar_t, 2>();

        if (n_dims == 2)
            execute_parallel_forward<scalar_t, 2>(matrices_a, eigenvectors_a, eigenvalues_a, n_batch);
        else
            execute_parallel_forward<scalar_t, 3>(matrices_a, eigenvectors_a, eigenvalues_a, n_batch);
    }));
    auto eigenvalues_shape = original_shape;
    eigenvalues_shape.pop_back();

    return {eigenvalues.view(eigenvalues_shape), eigenvectors.view(original_shape)};
}


//template <typename scalar_t, int DIMS>
//void execute_parallel_backward(const torch::PackedTensorAccessor32<scalar_t, 4>& mixture_a,
//                      const torch::PackedTensorAccessor32<scalar_t, 4>& xes_a,
//                      torch::PackedTensorAccessor32<scalar_t, 4>& grad_mixture_a,
//                      torch::PackedTensorAccessor32<scalar_t, 4>& grad_xes_a,
//                      torch::PackedTensorAccessor32<scalar_t, 3>& grad_output_a,
//                      const gm::MixtureAndXesNs& n, bool requires_grad_mixture, bool requires_grad_xes) {

//    const auto nXes_x_nLayers = int(n.xes * n.layers);
//    #pragma omp parallel for num_threads(16)
//    for (int i = 0; i < n.batch * n.layers * n.xes; ++i) {
//        const auto batch_index = int(i) / nXes_x_nLayers;
//        const auto remaining = (int(i) - batch_index * nXes_x_nLayers);
//        const auto layer_index = remaining / int(n.xes);
//        const auto batch_xes_index = std::min(batch_index, int(n.batch_xes - 1));
//        const auto layer_xes_index = std::min(layer_index, int(n.layers_xes - 1));
//        const auto xes_index = remaining - layer_index * int(n.xes);

//        const auto& x_pos = gm::vec<DIMS>(xes_a[batch_xes_index][layer_xes_index][xes_index][0]);

//        auto& grad_xes = gm::vec<DIMS>(grad_xes_a[batch_xes_index][layer_xes_index][xes_index][0]);
//        for (int c = 0; c < int(n.components); ++c) {
//            auto& grad_c_weight = gm::weight(grad_mixture_a[batch_index][layer_index][c]);
//            auto& grad_c_pos = gm::position<DIMS>(grad_mixture_a[batch_index][layer_index][c]);
//            auto& grad_c_cov = gm::covariance<DIMS>(grad_mixture_a[batch_index][layer_index][c]);

//            const auto& c_weight = gm::weight(mixture_a[batch_index][layer_index][c]);
//            const auto& c_pos = gm::position<DIMS>(mixture_a[batch_index][layer_index][c]);
//            const auto& c_cov = gm::covariance<DIMS>(mixture_a[batch_index][layer_index][c]);

//            const auto t = x_pos - c_pos;
//            const auto v = scalar_t(-0.5) * glm::dot(t, (c_cov * t));
//            const auto exp = std::exp(v);
//            const auto weighted_exp = c_weight * exp;
//            const auto local_grad_c_pos = weighted_exp * t * c_cov;

//            if (requires_grad_xes) {
//                // grad_xes causes a race in case of xes having only 1 batch or layer size
//                for (int i = 0; i < DIMS; ++i) {
//                    #pragma omp atomic
//                    grad_xes[i] += -local_grad_c_pos[i];
//                }
//            }
//            if (requires_grad_mixture) {
//                const auto weight_addition = exp * grad_output_a[batch_index][layer_index][xes_index];
//                const auto pos_addition = local_grad_c_pos * grad_output_a[batch_index][layer_index][xes_index];
//                const auto cov_addition = - c_weight * scalar_t(0.5) * exp * grad_output_a[batch_index][layer_index][xes_index] * glm::outerProduct(t, t);

//                #pragma omp atomic
//                grad_c_weight += weight_addition;

//                for (int i = 0; i < DIMS; ++i) {
//                    #pragma omp atomic
//                    grad_c_pos[i] += pos_addition[i];

//                    for (int j = 0; j < DIMS; ++j) {
//                        #pragma omp atomic
//                        grad_c_cov[i][j] += cov_addition[i][j];
//                    }
//                }
//            }
//        }
//        if (requires_grad_xes) {
//            for (int i = 0; i < DIMS; ++i) {
//                #pragma omp atomic
//                grad_xes[i] *= grad_output_a[batch_index][layer_index][xes_index];
//            }
//        }
//    }
//}

//std::vector<torch::Tensor> eigen_cpu_backward(torch::Tensor grad_output, torch::Tensor mixture, torch::Tensor xes, bool requires_grad_mixture, bool requires_grad_xes) {
//    gm::check_mixture(mixture);
//    auto n = gm::check_input_and_get_ns(mixture, xes);

//    TORCH_CHECK(mixture.device().is_cpu(), "this one is just for cpu..");
//    TORCH_CHECK(grad_output.device().is_cpu(), "grad_output must be on cpu..");
//    TORCH_CHECK(grad_output.dim() == 3, "grad_output has wrong number of dimensions");
//    TORCH_CHECK(grad_output.size(0) == n.batch, "grad_output has wrong batch dimension");
//    TORCH_CHECK(grad_output.size(1) == n.layers, "grad_output has wrong layer dimension");
//    TORCH_CHECK(grad_output.size(2) == n.xes, "grad_output has wrong xes dimension");
//    TORCH_CHECK(grad_output.dtype() == mixture.dtype(), "grad_output dtype does not match with mixture dtype")

//    torch::Tensor grad_mixture = torch::zeros({n.batch, n.layers, n.components, mixture.size(3)}, torch::dtype(mixture.dtype()).device(mixture.device()));
//    torch::Tensor grad_xes = torch::zeros({n.batch_xes, n.layers_xes, n.xes, n.dims}, torch::dtype(mixture.dtype()).device(mixture.device()));

//    AT_DISPATCH_FLOATING_TYPES(mixture.scalar_type(), "eval_inversed_omp_backward", ([&] {
//        auto mixture_a = mixture.packed_accessor32<scalar_t, 4>();
//        auto xes_a = xes.packed_accessor32<scalar_t, 4>();
//        auto grad_mixture_a = grad_mixture.packed_accessor32<scalar_t, 4>();
//        auto grad_xes_a = grad_xes.packed_accessor32<scalar_t, 4>();
//        auto grad_output_a = grad_output.packed_accessor32<scalar_t, 3>();

//        if (n.dims == 2)
//            execute_parallel_backward<scalar_t, 2>(mixture_a, xes_a, grad_mixture_a, grad_xes_a, grad_output_a, n, requires_grad_mixture, requires_grad_xes);
//        else
//            execute_parallel_backward<scalar_t, 3>(mixture_a, xes_a, grad_mixture_a, grad_xes_a, grad_output_a, n, requires_grad_mixture, requires_grad_xes);
//    }));

//    return {grad_mixture, grad_xes};
//}

#ifndef GMC_CMAKE_TEST_BUILD
PYBIND11_MODULE(symeig_cpu, m) {
  m.def("forward", &gpe::symeig_cpu_forward, "evaluate_inversed forward");
//  m.def("backward", &eigen_cpu_backward, "evaluate_inversed backward");
}
#endif
