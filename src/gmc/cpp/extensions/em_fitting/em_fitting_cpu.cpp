#include <torch/extension.h>

#include <vector>
#include <algorithm>


#include "common.h"
#include "em_fitting_common.h"


// preiner refers to "Continuous projection for fast L 1 reconstruction" (ACM TOG, 2014)

constexpr int N_VIRTUAL_POINTS = 100;

template <typename scalar_t, int DIMS>
void calc_likelihoods(const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits>& target_a,
                      const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits>& fitting_a,
                      torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits>& likelihoods_a,
                      const gm::MixtureNs& n,
                      const gm::MixtureNs& n_fitting) {

    const auto n_lcfc = int(n.layers * n.components * n_fitting.components);
    const auto n_cfc = int(n.components * n_fitting.components);

//    #pragma omp parallel for num_threads(16)
    for (int i = 0; i < int(n.batch * n.layers * n.components * n_fitting.components); ++i) {
        const auto batch_index = i / n_lcfc;
        const auto remaining = i - batch_index * n_lcfc;
        const auto layer_index = remaining / int(n_cfc);
        const auto remaining2 = remaining - layer_index * n_cfc;
        const auto target_component_index = remaining2 / int(n_fitting.components);
        const auto fitting_component_index = remaining2 - target_component_index * int(n_fitting.components);


        const auto& target_weight = gm::weight(target_a[batch_index][layer_index][target_component_index]);
        const auto& target_pos = gm::position<DIMS>(target_a[batch_index][layer_index][target_component_index]);
        const auto& target_cov = gm::covariance<DIMS>(target_a[batch_index][layer_index][target_component_index]);
        if (glm::determinant(target_cov) <= 0) {
            std::cout << "1 b" << batch_index << " l" << layer_index << " tc" << target_component_index << " fc" << fitting_component_index << std::endl;
            std::cout << "target_cov: [" << target_cov[0][0] << ", " << target_cov[0][1] << "; \n" << target_cov[1][0] << ", " << target_cov[1][1] << std::endl;
            std::cout << "target_cov det: " << glm::determinant(target_cov) << std::endl;
        }

        const auto& fitting_weight = gm::weight(fitting_a[batch_index][layer_index][fitting_component_index]);
        const auto& fitting_pos = gm::position<DIMS>(fitting_a[batch_index][layer_index][fitting_component_index]);
        const auto& fitting_cov = gm::covariance<DIMS>(fitting_a[batch_index][layer_index][fitting_component_index]);

        if (glm::determinant(target_cov) <= 0) {
            std::cout << "2 b" << batch_index << " l" << layer_index << " tc" << target_component_index << " fc" << fitting_component_index << std::endl;
            std::cout << "target_cov: [" << target_cov[0][0] << ", " << target_cov[0][1] << "; \n" << target_cov[1][0] << ", " << target_cov[1][1] << std::endl;
            std::cout << "target_cov det: " << glm::determinant(target_cov) << std::endl;
        }
        const auto& target_gaussian_amplitude = gaussian_amplitude<scalar_t, DIMS>(target_cov);
        const auto& target_n_virtual_points = scalar_t(N_VIRTUAL_POINTS) * target_weight / target_gaussian_amplitude;

        if (glm::determinant(fitting_cov) <= 0) {
            std::cout << "3 b" << batch_index << " l" << layer_index << " tc" << target_component_index << " fc" << fitting_component_index << std::endl;
            std::cout << "fitting_cov: [" << fitting_cov[0][0] << ", " << fitting_cov[0][1] << "; \n" << fitting_cov[1][0] << ", " << fitting_cov[1][1] << std::endl;
            std::cout << "fitting_cov det: " << glm::determinant(fitting_cov) << std::endl;
        }
        const auto& fitting_gaussian_amplitude = gaussian_amplitude<scalar_t, DIMS>(fitting_cov);

        // preiner Equation (9) and multiplied with w_s from Equation (8)
        const auto gaussianValue = gm::evaluate_gaussian(target_pos, fitting_gaussian_amplitude, fitting_pos, fitting_cov);
        const auto traceExp = gm::exp(scalar_t(-0.5) * trace(fitting_cov * glm::inverse(target_cov)));
        const auto likelihood = fitting_weight * std::pow(gaussianValue * traceExp, target_n_virtual_points);
        if (likelihood > 1000) {
            std::cout << "3 b" << batch_index << " l" << layer_index << " tc" << target_component_index << " fc" << fitting_component_index << std::endl;
            std::cout << "gaussianValue=" << gaussianValue << "  traceExp=" << traceExp << "  target_n_virtual_points=" << target_n_virtual_points << std::endl;
        }
        likelihoods_a[batch_index][layer_index][target_component_index][fitting_component_index] = likelihood;
    }
}

std::vector<torch::Tensor> forward(torch::Tensor target, torch::Tensor initial) {
    using namespace torch::indexing;
    gm::check_mixture(target);
    gm::check_mixture(initial);

    TORCH_CHECK(target.device().is_cpu(), "this one is just for cpu..");
    TORCH_CHECK(initial.device().is_cpu(), "this one is just for cpu..");
    auto n = gm::get_ns(target);
    auto n_fitting = gm::get_ns(initial);
    TORCH_CHECK(n.batch == n_fitting.batch);
    TORCH_CHECK(n.layers == n_fitting.layers);
    TORCH_CHECK(n.dims == n_fitting.dims);

    auto likelihoods = torch::zeros({n.batch, n.layers, n.components, n_fitting.components}, torch::dtype(target.dtype()).device(target.device()));

    AT_DISPATCH_FLOATING_TYPES(target.scalar_type(), "calc_likelihoods", ([&] {
        /// TODO: some of the covariances are inversed, that can be precomputed
        auto target_a = target.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>();
        auto fitting_a = initial.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>();
        auto likelihoods_a = likelihoods.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>();

        if (n.dims == 2)
            calc_likelihoods<scalar_t, 2>(target_a, fitting_a, likelihoods_a, n, n_fitting);
        else
            calc_likelihoods<scalar_t, 3>(target_a, fitting_a, likelihoods_a, n, n_fitting);
    }));
    return {likelihoods};
//    auto likelihoods_sum = likelihoods.sum({3}, true);
//    auto responsibilities = likelihoods / likelihoods_sum.where(likelihoods_sum > 0.00001, torch::ones_like(likelihoods_sum)); // preiner Equation (8)

//    // cont with preiner (10)
//    // index i -> target
//    // index s -> fitting
//    responsibilities = responsibilities * gm::weights(target).unsqueeze(-1);
//    const auto newWeights = torch::sum(responsibilities, {2});
//    responsibilities = responsibilities / newWeights.where(newWeights > 0.00001, torch::ones_like(newWeights)).view({n.batch, n.layers, 1, n_fitting.components});
//    const auto newPositions = torch::sum(responsibilities.unsqueeze(-1) * gm::positions(target).view({n.batch, n.layers, n.components, 1, n.dims}), {2});
//    const auto posDiffs = gm::positions(target).view({n.batch, n.layers, n.components, 1, n.dims, 1}) - newPositions.view({n.batch, n.layers, 1, n_fitting.components, n.dims, 1});

//    const auto newCovariances = (torch::sum(responsibilities.unsqueeze(-1).unsqueeze(-1) * (gm::covariances(target).inverse().view({n.batch, n.layers, n.components, 1, n.dims, n.dims}) +
//                                                               posDiffs.matmul(posDiffs.transpose(-1, -2))), {2}) + torch::eye(n.dims) * 0.001).inverse();

//    return {gm::pack_mixture(newWeights.contiguous(), newPositions.contiguous(), newCovariances.contiguous()), responsibilities};
}


//template <typename scalar_t, int DIMS>
//void execute_parallel_backward(const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits>& mixture_a,
//                      const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits>& xes_a,
//                      torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits>& grad_mixture_a,
//                      torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits>& grad_xes_a,
//                      torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>& grad_output_a,
//                      const gm::MixtureNs& n, bool requires_grad_mixture, bool requires_grad_xes) {

//    const auto nXes_x_nLayers = int(n.xes * n.layers);
//    #pragma omp parallel for num_threads(16)
//    for (uint i = 0; i < n.batch * n.layers * n.xes; ++i) {
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

std::vector<torch::Tensor> backward(torch::Tensor grad_output, torch::Tensor mixture, torch::Tensor xes, bool requires_grad_mixture, bool requires_grad_xes) {
    gm::check_mixture(mixture);
    auto n = gm::check_input_and_get_ns(mixture, xes);

    TORCH_CHECK(mixture.device().is_cpu(), "this one is just for cpu..");
    TORCH_CHECK(grad_output.device().is_cpu(), "grad_output must be on cpu..");
    TORCH_CHECK(grad_output.dim() == 3, "grad_output has wrong number of dimensions");
    TORCH_CHECK(grad_output.size(0) == n.batch, "grad_output has wrong batch dimension");
    TORCH_CHECK(grad_output.size(1) == n.layers, "grad_output has wrong layer dimension");
    TORCH_CHECK(grad_output.size(2) == n.xes, "grad_output has wrong xes dimension");
    TORCH_CHECK(grad_output.dtype() == mixture.dtype(), "grad_output dtype does not match with mixture dtype")

    torch::Tensor grad_mixture = torch::zeros({n.batch, n.layers, n.components, mixture.size(3)}, torch::dtype(mixture.dtype()).device(mixture.device()));
    torch::Tensor grad_xes = torch::zeros({n.batch_xes, n.layers_xes, n.xes, n.dims}, torch::dtype(mixture.dtype()).device(mixture.device()));

//    AT_DISPATCH_FLOATING_TYPES(mixture.scalar_type(), "eval_inversed_omp_backward", ([&] {
//        auto mixture_a = mixture.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>();
//        auto xes_a = xes.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>();
//        auto grad_mixture_a = grad_mixture.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>();
//        auto grad_xes_a = grad_xes.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>();
//        auto grad_output_a = grad_output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();

//        if (n.dims == 2)
//            execute_parallel_backward<scalar_t, 2>(mixture_a, xes_a, grad_mixture_a, grad_xes_a, grad_output_a, n, requires_grad_mixture, requires_grad_xes);
//        else
//            execute_parallel_backward<scalar_t, 3>(mixture_a, xes_a, grad_mixture_a, grad_xes_a, grad_output_a, n, requires_grad_mixture, requires_grad_xes);
//    }));

    return {grad_mixture, grad_xes};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "forward function");
  m.def("backward", &backward, "backward function");
}
