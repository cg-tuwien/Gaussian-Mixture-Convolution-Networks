#include <catch2/catch.hpp>
#include <torch/types.h>

#include "convolution_fitting/implementation_forward.h"
#include "convolution_fitting/implementation_backward.h"
#include "convolution_fitting/Tree.h"
#include "util/mixture.h"

template<unsigned N_DIMS>
torch::Tensor random_mixture(unsigned n_batch, unsigned n_channels, unsigned n_comps, double dx) {
    const auto double_dtype = torch::TensorOptions().dtype(torch::kFloat64);
    auto weights = torch::randn({n_batch, n_channels, n_comps}, double_dtype);
    weights = weights.where(weights.abs() < dx * 1.5, torch::ones_like(weights) * 1.5 * dx);
    const auto positions = torch::randn({n_batch, n_channels, n_comps, N_DIMS}, double_dtype);
    const auto covs_p = torch::randn({n_batch, n_channels, n_comps, N_DIMS, N_DIMS}, double_dtype);
    return gpe::pack_mixture(weights, positions, covs_p.matmul(covs_p.transpose(-1, -2)) + torch::eye(N_DIMS, double_dtype).view({1, 1, 1, N_DIMS, N_DIMS}) * 0.05);
}

TEST_CASE("convolution_fitting") {
    using namespace torch::indexing;
    constexpr auto N_BATCH = 1;
    constexpr auto N_IN_CHANNELS = 1;
    constexpr auto N_OUT_CHANNELS = 1;
    constexpr auto N_DATA_COMPONENTS = 1;
    constexpr auto N_KERNEL_COMPONENTS = 1;
    constexpr auto N_FITTING_COMPONENTS = 1;
    constexpr auto N_DIMS = 2;
    constexpr auto DX = 0.00000001;
    using scalar_t = double;

    const auto double_dtype = torch::TensorOptions().dtype(torch::kFloat64);
    const auto data = random_mixture<N_DIMS>(N_BATCH, N_IN_CHANNELS, N_DATA_COMPONENTS, DX);
    const auto kernels = random_mixture<N_DIMS>(N_OUT_CHANNELS, N_IN_CHANNELS, N_KERNEL_COMPONENTS, DX);

    SECTION("jacobian") {
        using Tree = convolution_fitting::Tree<scalar_t, N_DIMS>;
        convolution_fitting::Config config{N_FITTING_COMPONENTS};

        typename Tree::Data tree_data_storage;
        Tree tree(data, kernels, &tree_data_storage, config);
        tree.create_tree_nodes();
        tree.create_attributes();
        tree.select_fitting_subtrees();

        // f: R^(M + N) -> R^O
        auto out = convolution_fitting::forward_with_given_tree<1, scalar_t, N_DIMS>(config, tree);
        out.data = data;
        out.kernels = kernels;

        const auto M = data.numel();
        const auto N = kernels.numel();
        const auto O = out.fitting.numel();

        auto analytical_jacobi = torch::zeros({O, M + N}, double_dtype);
        const auto grad = torch::eye(O, double_dtype);
        for (auto i = 0; i < O; ++i) {
            const auto grad_i = grad[i].view_as(out.fitting);
            torch::Tensor jacobi_i_data, jacobi_i_kernels;
            std::tie(jacobi_i_data, jacobi_i_kernels) = convolution_fitting::backward_impl_t<1, scalar_t, N_DIMS>(grad_i, out, config);
            const auto jacobi_i = torch::cat({jacobi_i_data.view(-1), jacobi_i_kernels.view(-1)}, 0);
            analytical_jacobi.index_put_({i, Slice()}, jacobi_i);
        }

        auto numerical_jacobi = torch::zeros({O, M + N}, double_dtype);
        const auto dx = torch::eye(M + N, double_dtype) * DX;
        for (auto j = 0; j < M + N; ++j) {
            std::cout << "j: " << j << std::endl;
            const auto dx_i_data = dx.index({j, Slice(0, M)}).view_as(data);
            const auto dx_i_kernels = dx.index({j, Slice(M, M + N)}).view_as(kernels);
            std::cout << "dx_i_data: " << dx_i_data << std::endl;
            std::cout << "dx_i_kernels: " << dx_i_kernels << std::endl;

            const auto data_plus = data + dx_i_data;
            const auto data_minus = data - dx_i_data;
            std::cout << "data_plus: " << data_plus << std::endl;
            std::cout << "data_minus: " << data_minus << std::endl;

            const auto kernels_plus = kernels + dx_i_kernels;
            const auto kernels_minus = kernels - dx_i_kernels;
            std::cout << "kernels_plus: " << kernels_plus << std::endl;
            std::cout << "kernels_minus: " << kernels_minus << std::endl;

            typename Tree::Data plus_storage;
            Tree plus_tree(data + dx_i_data, kernels + dx_i_kernels, &plus_storage, config);
            plus_tree.set_nodes_and_friends(tree_data_storage.nodes, tree_data_storage.node_attributes, tree_data_storage.fitting_subtrees);
            typename Tree::Data minus_storage;
            Tree minus_tree(data - dx_i_data, kernels - dx_i_kernels, &plus_storage, config);
            minus_tree.set_nodes_and_friends(tree_data_storage.nodes, tree_data_storage.node_attributes, tree_data_storage.fitting_subtrees);


            const auto out_plus_dx = convolution_fitting::forward_with_given_tree<1, scalar_t, N_DIMS>(config, plus_tree).fitting;
            const auto out_minus_dx = convolution_fitting::forward_with_given_tree<1, scalar_t, N_DIMS>(config, minus_tree).fitting;
            std::cout << "out_plus_dx: " << out_plus_dx << std::endl;
            std::cout << "out_minus_dx: " << out_minus_dx << std::endl;

            const auto jacobi_j = (out_plus_dx - out_minus_dx) / (2 * DX);
            numerical_jacobi.index_put_({Slice(), j}, jacobi_j.view(-1));
        }
        std::cout << "numerical: " << numerical_jacobi << std::endl;
        std::cout << "analytical: " <<  analytical_jacobi << std::endl;

        REQUIRE((numerical_jacobi - analytical_jacobi).square().mean().item<scalar_t>() < scalar_t(0.00000000001));
    }
}
