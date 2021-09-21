#include <catch2/catch.hpp>
#include <torch/types.h>

#include "evaluate_inversed/implementations.h"
#include "util/mixture.h"

template<unsigned N_DIMS>
torch::Tensor random_mixture(unsigned n_batch, unsigned n_channels, unsigned n_comps, double dx) {
    const auto double_dtype = torch::TensorOptions().dtype(torch::kFloat64);
    auto weights = torch::randn({n_batch, n_channels, n_comps}, double_dtype);
    weights = weights.where(weights.abs() >= dx * 1.5, torch::ones_like(weights) * 1.5 * dx);   // prevent sign change due to dx
    weights = weights.where(weights.abs() >= 0.1, torch::ones_like(weights) * 0.1);           // it's necessary to divide the gradient by the weight, sometimes we are unlucky and get a convolution weight of min*min, so this prevents exploding gradients.
    const auto positions = torch::randn({n_batch, n_channels, n_comps, N_DIMS}, double_dtype);
    const auto covs_p = torch::randn({n_batch, n_channels, n_comps, N_DIMS, N_DIMS}, double_dtype);
    return gpe::pack_mixture(weights, positions, covs_p.matmul(covs_p.transpose(-1, -2)) + torch::eye(N_DIMS, double_dtype).view({1, 1, 1, N_DIMS, N_DIMS}) * (dx * 10 + 0.05));
}

template<unsigned N_DIMS>
double grad_check(double dx, const unsigned n_batch, const unsigned n_channels, const unsigned n_components, const unsigned n_xes) {
    using namespace torch::indexing;

    using scalar_t = double;

    const auto double_dtype = torch::TensorOptions().dtype(torch::kFloat64);
    const auto mixture = random_mixture<N_DIMS>(n_batch, n_channels, n_components, dx);
    const auto xes = torch::randn({n_batch, n_channels, n_xes, N_DIMS}, double_dtype) * scalar_t(2.0);

    // f: R^(M + N) -> R^O
    auto out = parallel_forward_impl(mixture, xes);

    const auto M = mixture.numel();
    const auto N = xes.numel();
    const auto O = out.numel();

    auto analytical_jacobi = torch::zeros({O, M + N}, double_dtype);
    const auto grad = torch::eye(O, double_dtype);
    for (auto i = 0; i < O; ++i) {
        const auto grad_i = grad[i].view_as(out);
        torch::Tensor jacobi_i_mixture, jacobi_i_xes;
        std::tie(jacobi_i_mixture, jacobi_i_xes) = parallel_backward_impl(grad_i, mixture, xes, true, true);
        const auto jacobi_i = torch::cat({jacobi_i_mixture.view(-1), jacobi_i_xes.view(-1)}, 0);
        analytical_jacobi.index_put_({i, Slice()}, jacobi_i);
    }

    auto numerical_jacobi = torch::zeros({O, M + N}, double_dtype);
    const auto dx_mat = torch::eye(M + N, double_dtype) * dx;
    for (auto j = 0; j < M + N; ++j) {
        const auto dx_i_mixture = dx_mat.index({j, Slice(0, M)}).view_as(mixture);
        const auto dx_i_xes = dx_mat.index({j, Slice(M, M + N)}).view_as(xes);

        const auto mixture_plus = mixture + dx_i_mixture;
        const auto mixture_minus = mixture - dx_i_mixture;

        const auto xes_plus = xes + dx_i_xes;
        const auto xes_minus = xes - dx_i_xes;

        const auto out_plus_dx = parallel_forward_impl(mixture_plus, xes_plus);
        const auto out_minus_dx = parallel_forward_impl(mixture_minus, xes_minus);

        const auto jacobi_j = (out_plus_dx - out_minus_dx) / (2 * dx);
        numerical_jacobi.index_put_({Slice(), j}, jacobi_j.view(-1));
    }
//    std::cout << "numerical: " << numerical_jacobi << std::endl;
//    std::cout << "analytical: " <<  analytical_jacobi << std::endl;
    const auto max_error = (numerical_jacobi - analytical_jacobi).abs().max().template item<scalar_t>();
    return max_error;
}


TEST_CASE("grad_evaluate_inversed") {
    torch::manual_seed(0);
//    const auto dx_set = {0.0000001};
    const auto dx = 1e-08;  // tested, gives lowest error

    // full test:
//    const auto n_in_channels_set = {1, 2, 3, 16};
//    const auto n_data_components_set = {1, 2, 5, 8, 32, 64};
//    const auto n_kernel_components_set = {1, 2, 5};
//    const auto n_fitting_components_set = {2, 4, 32, 64};
    // quick test:
//    const auto n_batch_set = {1, 4};
//    const auto n_channels_set = {1, 5};
//    const auto n_components_set = {1, 6};
//    const auto n_xes_set = {1, 7};
    // quick test:
    const auto n_batch_set = {1};
    const auto n_channels_set = {1};
    const auto n_components_set = {1};
    const auto n_xes_set = {1};


    SECTION("jacobian") {
        for (const auto n_batch : n_batch_set) {
            for (const auto n_channels : n_channels_set) {
                for (const auto n_components : n_components_set) {
                    for (const auto n_xes : n_xes_set) {
                        REQUIRE(grad_check<2>(dx, unsigned(n_batch), unsigned(n_channels), unsigned(n_components), unsigned(n_xes)) < 2e-07);
                        REQUIRE(grad_check<3>(dx, unsigned(n_batch), unsigned(n_channels), unsigned(n_components), unsigned(n_xes)) < 2e-07);
                    }
                }
            }
        }
    }
}
