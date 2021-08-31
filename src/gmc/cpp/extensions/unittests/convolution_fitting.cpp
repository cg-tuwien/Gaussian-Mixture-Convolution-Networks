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
    weights = weights.where(weights.abs() >= dx * 1.5, torch::ones_like(weights) * 1.5 * dx);   // prevent sign change due to dx
    weights = weights.where(weights.abs() >= 0.1, torch::ones_like(weights) * 0.1);           // it's necessary to divide the gradient by the weight, sometimes we are unlucky and get a convolution weight of min*min, so this prevents exploding gradients.
    const auto positions = torch::randn({n_batch, n_channels, n_comps, N_DIMS}, double_dtype);
    const auto covs_p = torch::randn({n_batch, n_channels, n_comps, N_DIMS, N_DIMS}, double_dtype);
    return gpe::pack_mixture(weights, positions, covs_p.matmul(covs_p.transpose(-1, -2)) + torch::eye(N_DIMS, double_dtype).view({1, 1, 1, N_DIMS, N_DIMS}) * (dx * 10 + 0.05));
}

template<unsigned N_DIMS>
double grad_check(double dx, const unsigned n_in_channels, const unsigned n_data_components, const unsigned n_kernel_components, const unsigned n_fitting_components) {
    using namespace torch::indexing;
    constexpr auto n_batch = 3;
    const unsigned n_out_channels = 2;

    using scalar_t = double;

    const auto double_dtype = torch::TensorOptions().dtype(torch::kFloat64);
    const auto data = random_mixture<N_DIMS>(n_batch, n_in_channels, n_data_components, dx);
    const auto kernels = random_mixture<N_DIMS>(n_out_channels, n_in_channels, n_kernel_components, dx);
    using Tree = convolution_fitting::Tree<scalar_t, N_DIMS>;
    convolution_fitting::Config config{n_fitting_components};

    typename Tree::Data tree_data_storage;
    Tree tree(data, kernels, &tree_data_storage, config);
    tree.create_tree_nodes();
    tree.create_attributes();
    tree.select_fitting_subtrees();

    // f: R^(M + N) -> R^O
    auto out = convolution_fitting::forward_with_given_tree<1, scalar_t, N_DIMS>(config, tree);

    const auto M = data.numel();
    const auto N = kernels.numel();
    const auto O = out.fitting.numel();

    auto analytical_jacobi = torch::zeros({O, M + N}, double_dtype);
    const auto grad = torch::eye(O, double_dtype);
    for (auto i = 0; i < O; ++i) {
        const auto grad_i = grad[i].view_as(out.fitting);
        torch::Tensor jacobi_i_data, jacobi_i_kernels;
        std::tie(jacobi_i_data, jacobi_i_kernels) = convolution_fitting::backward_impl_t<1, scalar_t, N_DIMS>(grad_i, data, kernels, out, config);
        const auto jacobi_i = torch::cat({jacobi_i_data.view(-1), jacobi_i_kernels.view(-1)}, 0);
        analytical_jacobi.index_put_({i, Slice()}, jacobi_i);
    }

    auto numerical_jacobi = torch::zeros({O, M + N}, double_dtype);
    const auto dx_mat = torch::eye(M + N, double_dtype) * dx;
    for (auto j = 0; j < M + N; ++j) {
        const auto dx_i_data = dx_mat.index({j, Slice(0, M)}).view_as(data);
        const auto dx_i_kernels = dx_mat.index({j, Slice(M, M + N)}).view_as(kernels);

        const auto data_plus = data + dx_i_data;
        const auto data_minus = data - dx_i_data;

        const auto kernels_plus = kernels + dx_i_kernels;
        const auto kernels_minus = kernels - dx_i_kernels;

        typename Tree::Data plus_storage;
        Tree plus_tree(data + dx_i_data, kernels + dx_i_kernels, &plus_storage, config);
        plus_tree.set_nodes_and_friends(tree_data_storage.nodes, tree_data_storage.node_attributes, tree_data_storage.fitting_subtrees);
        typename Tree::Data minus_storage;
        Tree minus_tree(data - dx_i_data, kernels - dx_i_kernels, &minus_storage, config);
        minus_tree.set_nodes_and_friends(tree_data_storage.nodes, tree_data_storage.node_attributes, tree_data_storage.fitting_subtrees);


        const auto out_plus_dx = convolution_fitting::forward_with_given_tree<1, scalar_t, N_DIMS>(config, plus_tree).fitting;
        const auto out_minus_dx = convolution_fitting::forward_with_given_tree<1, scalar_t, N_DIMS>(config, minus_tree).fitting;

        const auto jacobi_j = (out_plus_dx - out_minus_dx) / (2 * dx);
        numerical_jacobi.index_put_({Slice(), j}, jacobi_j.view(-1));
    }
//        std::cout << "numerical: " << numerical_jacobi << std::endl;
//        std::cout << "analytical: " <<  analytical_jacobi << std::endl;
    const auto max_error = (numerical_jacobi - analytical_jacobi).abs().max().template item<scalar_t>();
//    std::cout << "n_in_channels: " << n_in_channels << ", n_data_components: " << n_data_components << ", n_kernel_components: " << n_kernel_components << ", n_fitting_components: " << n_fitting_components << ", dx: " << dx << ", max_error: " <<  max_error << std::endl;
    return max_error;
}


TEST_CASE("convolution_fitting") {
//    const auto dx_set = {0.0000001};
    const auto dx = 1e-07;  // tested, gives lowest error

    // full test:
//    const auto n_in_channels_set = {1, 2, 3, 16};
//    const auto n_data_components_set = {1, 2, 5, 8, 32, 64};
//    const auto n_kernel_components_set = {1, 2, 5};
//    const auto n_fitting_components_set = {2, 4, 32, 64};
    // quick test:
    const auto n_in_channels_set = {1, 4};
    const auto n_data_components_set = {1, 8};
    const auto n_kernel_components_set = {5};
    const auto n_fitting_components_set = {2, 32};


    SECTION("jacobian") {
        for (const auto n_in_channels : n_in_channels_set) {
            for (const auto n_data_components : n_data_components_set) {
                for (const auto n_kernel_components : n_kernel_components_set) {
                    for (const auto n_fitting_components : n_fitting_components_set) {
                        REQUIRE(grad_check<2>(dx, unsigned(n_in_channels), unsigned(n_data_components), unsigned(n_kernel_components), unsigned(n_fitting_components)) < 2e-07);
                        REQUIRE(grad_check<3>(dx, unsigned(n_in_channels), unsigned(n_data_components), unsigned(n_kernel_components), unsigned(n_fitting_components)) < 2e-07);
                    }
                }
            }
        }
    }
}
