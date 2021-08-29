#include "implementation_autodiff_backward.h"
#include "implementation_forward.h"

#include "util/autodiff.h"

namespace bvh_mhem_fit_alpha {

template<int REDUCTION_N, typename scalar_t, unsigned N_DIMS>
ForwardBackWardOutput implementation_autodiff_backward(torch::Tensor mixture, const torch::Tensor& gradient_fitting, const Config& config) {
    using namespace torch::indexing;
    using LBVH = lbvh::Bvh<N_DIMS, scalar_t>;
    using AutoDiffScalar = autodiff::Variable<scalar_t>;
    using AutoDiffGaussian = gpe::Gaussian<N_DIMS, AutoDiffScalar>;

    // todo: flatten mixture for kernel, i.g. nbatch/nlayers/ncomponents/7 => nmixture/ncomponents/7

    auto n = gpe::get_ns(mixture);
    TORCH_CHECK(n.batch * n.layers < 65535, "n_batch x n_layers must be smaller than 65535 for CUDA")
    TORCH_CHECK(n.components > 1, "number of components must be greater 1 for this implementation")
    TORCH_CHECK(n.components < 65535, "number of components must be smaller than 65535 for morton code computation")
    TORCH_CHECK(n.dims == N_DIMS, "something wrong with dispatch")
    TORCH_CHECK(mixture.dtype() == caffe2::TypeMeta::Make<scalar_t>(), "something wrong with dispatch, or maybe this float type is not supported.")

    const auto n_mixtures = n.batch * n.layers;
    const LBVH bvh = LBVH(gpe::mixture_with_inversed_covariances(mixture).contiguous(), config.bvh_config);

    const auto n_internal_nodes = bvh.m_n_internal_nodes;
    const auto n_nodes = bvh.m_n_nodes;
    mixture = mixture.view({n_mixtures, n.components, -1}).contiguous();
    auto flat_bvh_nodes = bvh.m_nodes.view({n_mixtures, n_nodes, -1});
    auto flat_bvh_aabbs = bvh.m_aabbs.view({n_mixtures, n_nodes, -1});
    auto flag_container = torch::zeros({n_mixtures, n_internal_nodes}, torch::TensorOptions(mixture.device()).dtype(torch::ScalarType::Int));
    auto node_attributes = torch::zeros({n_mixtures, n_nodes, sizeof(typename AugmentedBvh<scalar_t, N_DIMS, REDUCTION_N>::NodeAttributes)}, torch::TensorOptions(mixture.device()).dtype(torch::ScalarType::Byte));

    std::vector<AutoDiffGaussian> mixture_autodiff;
    {
        auto mixture_a = gpe::accessor<scalar_t, 3>(mixture);
        mixture_autodiff.reserve(size_t(n_mixtures * n.components));
        for (int i = 0; i < mixture.size(0); ++i) {
            for (int j = 0; j < mixture.size(1); ++j) {
                mixture_autodiff.push_back(AutoDiffGaussian(gpe::gaussian<N_DIMS>(mixture_a[i][j])));
            }
        }
    }
    std::vector<AutoDiffScalar> aabbs_autodiff;
    {
        const auto aabbs_a = gpe::accessor<scalar_t, 3>(flat_bvh_aabbs);
        aabbs_autodiff.reserve(size_t(flat_bvh_aabbs.size(0) * flat_bvh_aabbs.size(1) * flat_bvh_aabbs.size(2)));
        for (int i = 0; i < flat_bvh_aabbs.size(0); ++i) {
            for (int j = 0; j < flat_bvh_aabbs.size(1); ++j) {
                for (int k = 0; k < flat_bvh_aabbs.size(2); ++k) {
                    aabbs_autodiff.push_back(AutoDiffScalar(aabbs_a[i][j][k]));
                }
            }
        }
    }
    std::vector<typename AugmentedBvh<AutoDiffScalar, N_DIMS, REDUCTION_N>::NodeAttributes> nodes_autodiff(unsigned(n_mixtures) * n_nodes);


    const auto mixture_a = gpe::accessor<AutoDiffGaussian, 2>(mixture_autodiff, {uint32_t(n_mixtures), uint32_t(n.components)});
    const auto aabbs_a = gpe::accessor<AutoDiffScalar, 3>(aabbs_autodiff, {uint32_t(flat_bvh_aabbs.size(0)), uint32_t(flat_bvh_aabbs.size(1)), uint32_t(flat_bvh_aabbs.size(2))});
    const auto nodes_a = gpe::accessor<lbvh::detail::Node::index_type_torch, 3>(flat_bvh_nodes);
    auto flags_a = gpe::accessor<int, 2>(flag_container);
    auto node_attributes_a = gpe::accessor<typename AugmentedBvh<AutoDiffScalar, N_DIMS, REDUCTION_N>::NodeAttributes, 2>(nodes_autodiff, {uint32_t(n_mixtures), uint32_t(n_nodes)});
    {
        dim3 dimBlock = dim3(32, 1, 1);
        dim3 dimGrid = dim3(uint(1),
                            (uint(n_mixtures) + dimBlock.y - 1) / dimBlock.y,
                            (uint(1) + dimBlock.z - 1) / dimBlock.z);

        auto fun = [mixture_a, nodes_a, aabbs_a, flags_a, node_attributes_a, n, n_mixtures, n_internal_nodes, n_nodes, config]
                (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
            iterate_over_nodes<AutoDiffScalar, N_DIMS, REDUCTION_N>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx,
                                                              mixture_a, nodes_a, aabbs_a, flags_a, node_attributes_a,
                                                              n, n_mixtures, n_internal_nodes, n_nodes,
                                                              config);
        };
        gpe::start_serial(gpe::device(mixture), dimGrid, dimBlock, fun);
    }

//    auto out_mixture = torch::zeros({n_mixtures, config.n_components_fitting, mixture.size(-1)}, torch::TensorOptions(mixture.device()).dtype(mixture.dtype()));
    std::vector<AutoDiffGaussian> out_mixture;
    out_mixture.resize(unsigned(n_mixtures) * config.n_components_fitting);
    auto out_mixture_a = gpe::accessor<AutoDiffGaussian, 2>(out_mixture, {uint32_t(n_mixtures), uint32_t(config.n_components_fitting)});

    // make it valid, in case something doesn't get filled (due to an inbalance of the tree or just not enough elements)
    {
        dim3 dimBlock = dim3(32, 1, 1);
        dim3 dimGrid = dim3((uint(n_mixtures) + dimBlock.x - 1) / dimBlock.x, 1, 1);

        auto fun = [mixture_a, out_mixture_a, nodes_a, aabbs_a, flags_a, node_attributes_a, n, n_mixtures, n_internal_nodes, n_nodes, config]
                (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
            collect_result<AutoDiffScalar, N_DIMS, REDUCTION_N>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx,
                                                          mixture_a, out_mixture_a, nodes_a, aabbs_a, flags_a, node_attributes_a,
                                                          n, n_mixtures, n_internal_nodes, n_nodes,
                                                          config);
        };
        gpe::start_serial(gpe::device(mixture), dimGrid, dimBlock, fun);
    }

    // set gradients to 0
    for(AutoDiffGaussian& g : out_mixture) {
        auto& w = g.weight;
        auto& p = g.position;
        auto& c = g.covariance;
        w.seed();
        for (int i = 0; i < int(N_DIMS); ++i) {
            p[i].seed();
            for (int j = 0; j < int(N_DIMS); ++j) {
                c[i][j].seed();
            }
        }
    }
    // backprop
    {
        auto gradient = gradient_fitting.view({n_mixtures * int(config.n_components_fitting), -1});
        auto gradient_a = gpe::struct_accessor<gpe::Gaussian<N_DIMS, scalar_t>, 1, scalar_t>(gradient);
        int g_index = 0;
        for(AutoDiffGaussian& g : out_mixture) {
            auto& w = g.weight;
            auto& p = g.position;
            auto& c = g.covariance;
            w.expr->propagate(gradient_a[g_index].weight);
            for (int i = 0; i < int(N_DIMS); ++i) {
                auto grad_p = gradient_a[g_index].position[i];
                p[i].expr->propagate(grad_p);
                for (int j = 0; j < int(N_DIMS); ++j) {
                    c[i][j].expr->propagate(gradient_a[g_index].covariance[i][j]);
                }
            }
            ++g_index;
        }
    }

    // extract
    ForwardBackWardOutput out;
    out.output = torch::zeros({n_mixtures * int(config.n_components_fitting), mixture.size(-1)}, torch::TensorOptions(mixture.dtype()));
    auto output_a = gpe::accessor<scalar_t, 2>(out.output);
    int index = 0;
    for(const AutoDiffGaussian& g : out_mixture) {
        output_a[index][0] = autodiff::val(g.weight);
        for (int i = 0; i < int(N_DIMS); ++i) {
            output_a[index][i+1] = autodiff::val(g.position[i]);
            for (int j = 0; j < int(N_DIMS); ++j) {
                output_a[index][1 + int(N_DIMS) + i * int(N_DIMS) + j] = autodiff::val(g.covariance[i][j]);
            }
        }
        index++;
    }
    out.output = out.output.view({n.batch, n.layers, config.n_components_fitting, -1});

    out.mixture_gradient = torch::zeros({n_mixtures * n.components, mixture.size(-1)}, torch::TensorOptions(mixture.dtype()));
    auto mixture_gradient_a = gpe::accessor<scalar_t, 2>(out.mixture_gradient);
    index = 0;
    for(const AutoDiffGaussian& g : mixture_autodiff) {
        mixture_gradient_a[index][0] = g.weight.grad();
        for (int i = 0; i < int(N_DIMS); ++i) {
            mixture_gradient_a[index][i+1] = g.position[i].grad();
            for (int j = 0; j < int(N_DIMS); ++j) {
                mixture_gradient_a[index][1 + int(N_DIMS) + i * int(N_DIMS) + j] = g.covariance[i][j].grad();
            }
        }
        index++;
    }
    out.mixture_gradient = out.mixture_gradient.view({n.batch, n.layers, n.components, -1});

    return out;
}

template ForwardBackWardOutput implementation_autodiff_backward<2, float, 2>(torch::Tensor mixture, const torch::Tensor& gradient_fitting, const Config& config);
template ForwardBackWardOutput implementation_autodiff_backward<2, double, 2>(torch::Tensor mixture, const torch::Tensor& gradient_fitting, const Config& config);
template ForwardBackWardOutput implementation_autodiff_backward<4, float, 2>(torch::Tensor mixture, const torch::Tensor& gradient_fitting, const Config& config);
template ForwardBackWardOutput implementation_autodiff_backward<4, double, 2>(torch::Tensor mixture, const torch::Tensor& gradient_fitting, const Config& config);
template ForwardBackWardOutput implementation_autodiff_backward<8, float, 2>(torch::Tensor mixture, const torch::Tensor& gradient_fitting, const Config& config);
template ForwardBackWardOutput implementation_autodiff_backward<8, double, 2>(torch::Tensor mixture, const torch::Tensor& gradient_fitting, const Config& config);

}
