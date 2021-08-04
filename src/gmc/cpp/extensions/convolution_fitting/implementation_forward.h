#include "convolution_fitting/implementation.h"
#include <stdio.h>
#include <type_traits>

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include "convolution_fitting/implementation_common.h"
#include "convolution_fitting/Config.h"
#include "convolution_fitting/Tree.h"
#include "common.h"
#include "cuda_qt_creator_definitinos.h"
#include "cuda_operations.h"
#include "hacked_accessor.h"
#include "util/glm.h"
#include "util/scalar.h"
#include "util/algorithms.h"
#include "util/autodiff.h"
#include "util/containers.h"
#include "util/cuda.h"
#include "util/gaussian.h"
#include "util/gaussian_mixture.h"
#include "util/helper.h"
#include "util/mixture.h"
#include "parallel_start.h"


namespace convolution_fitting {

namespace  {



} // anonymous namespace


template<int REDUCTION_N = 4, typename scalar_t, unsigned N_DIMS>
ForwardOutput forward_impl_t(const torch::Tensor& data, const torch::Tensor& kernels, const Config& config) {
    // cont with fitting todos:
    // - fetch positions + compute morton codes.
    // - pack and store everything in the morton code array
    // - sort that array and create tree structure
    // - store integrals in leaves (consider array of structs for gaussian data)
    // - bottom up pass for computing integral sums and n_leaves
    // - top down pass to select start nodes
    // - implement simple fitting (1G) -> test performance against TreeHem
    // - if 1G bad, implement 2G and 4G fitting.
    //
    // cont with gradient computation
    // - gradient for fitting alone (tests only)
    // - merge and test. [we have no trickle down now, yey.]


    const auto n = gpe::get_ns(data);
    const auto kernel_n = gpe::get_ns(kernels);
    const auto n_channels_in = n.layers;
    const auto n_channels_out = kernel_n.batch;
    const auto n_target_components = unsigned(n.components * n_channels_in * kernel_n.components);
    TORCH_CHECK(n.batch * n_channels_out < 65535, "n_batch x n_layers must be smaller than 65535 for CUDA")
    TORCH_CHECK(n.components >= 1, "number of components must be greater 1 for this implementation")
    TORCH_CHECK(kernel_n.components >= 1, "number of components must be greater 1 for this implementation")
    TORCH_CHECK(n_channels_in == kernel_n.layers, "number of input feature maps must agree with the second kernel dimension")
    TORCH_CHECK(n.dims == kernel_n.dims, "number of dimensions of data and kernel must agree")
    TORCH_CHECK(n.dims == N_DIMS, "something wrong with dispatch")
    TORCH_CHECK(data.dtype() == kernels.dtype(), "kernel and data dtypes must agree")
    TORCH_CHECK(data.dtype() == caffe2::TypeMeta::Make<scalar_t>(), "something wrong with dispatch, or maybe this float type is not supported.")
    TORCH_CHECK(data.device() == kernels.device(), "data and kernel devices must agree")

    const auto data_a = gpe::struct_accessor<typename gpe::Gaussian<N_DIMS, scalar_t>, 3, scalar_t>(data);
    const auto kernel_a = gpe::struct_accessor<typename gpe::Gaussian<N_DIMS, scalar_t>, 3, scalar_t>(kernels);

    Tree<scalar_t, N_DIMS> tree(data, kernels, config);


//    const auto n_mixtures = n.batch * n.layers;
//    auto bvh_config = config.bvh_config;
//    bvh_config.make_aabbs = false;
//    const Tree bvh = Tree(gpe::mixture_with_inversed_covariances(mixture).contiguous(), bvh_config);
//    const auto n_internal_nodes = bvh.m_n_internal_nodes;
//    const auto n_nodes = bvh.m_n_nodes;
//    mixture = mixture.view({n_mixtures, n.components, -1}).contiguous();
//    auto flat_bvh_nodes = bvh.m_nodes.view({n_mixtures, n_nodes, -1});
//    auto flag_container = torch::zeros({n_mixtures, n_internal_nodes}, torch::TensorOptions(mixture.device()).dtype(torch::ScalarType::Int));

//    auto flags_a = gpe::accessor<int, 2>(flag_container);
//    auto node_attributes = torch::zeros({n_mixtures, n_nodes, sizeof(typename AugmentedBvh<scalar_t, N_DIMS, REDUCTION_N>::NodeAttributes)}, torch::TensorOptions(mixture.device()).dtype(torch::ScalarType::Byte));



    auto out_mixture = torch::empty({n.batch, n_channels_out, n_target_components, data.size(-1)}, torch::TensorOptions(data.device()).dtype(data.dtype()));
//    auto out_mixture_a = gpe::struct_accessor<gpe::Gaussian<N_DIMS, scalar_t>, 3, scalar_t>(out_mixture);

//    std::cout << "n_target_components: " << n_target_components << std::endl;
//    std::cout << "n.batch: " << n.batch << std::endl;
//    std::cout << "n_channels_out: " << n_channels_out << std::endl;
//    std::cout << "n_channels_in: " << n_channels_in << std::endl;
//    std::cout << "kernel_n.components: " << kernel_n.components << std::endl;
//    std::cout << "n.components: " << n.components << std::endl;







//    auto nodes_a = gpe::accessor<lbvh::detail::Node::index_type_torch, 3>(flat_bvh_nodes);
//    auto node_attributes_a = gpe::struct_accessor<typename AugmentedBvh<scalar_t, N_DIMS, REDUCTION_N>::NodeAttributes, 2, uint8_t>(node_attributes);

//    {
//        dim3 dimBlock = dim3(32, 1, 1);
//        dim3 dimGrid = dim3(uint(1),
//                            (uint(n_mixtures) + dimBlock.y - 1) / dimBlock.y,
//                            (uint(1) + dimBlock.z - 1) / dimBlock.z);

//        auto fun = [mixture_a, nodes_a, flags_a, node_attributes_a, n, n_mixtures, n_internal_nodes, n_nodes, config] __host__ __device__
//                (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
//            iterate_over_nodes<scalar_t, N_DIMS, REDUCTION_N>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx,
//                                                              mixture_a, nodes_a, flags_a, node_attributes_a,
//                                                              n, n_mixtures, n_internal_nodes, n_nodes,
//                                                              config);
//        };
//        gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(mixture), dimGrid, dimBlock, fun);
//    }

//    auto out_mixture = torch::empty({n_mixtures, config.n_components_fitting, mixture.size(-1)}, torch::TensorOptions(mixture.device()).dtype(mixture.dtype()));
//    auto out_mixture_a = gpe::struct_accessor<gpe::Gaussian<N_DIMS, scalar_t>, 2, scalar_t>(out_mixture);

//    // make it valid, in case something doesn't get filled (due to an inbalance of the tree or just not enough elements)
//    {
//        dim3 dimBlock = dim3(32, 1, 1);
//        dim3 dimGrid = dim3((uint(n_mixtures) + dimBlock.x - 1) / dimBlock.x, 1, 1);

//        auto fun = [mixture_a, out_mixture_a, nodes_a, flags_a, node_attributes_a, n, n_mixtures, n_internal_nodes, n_nodes, config]
//                __host__ __device__
//                (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
//            collect_result<scalar_t, N_DIMS, REDUCTION_N>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx,
//                                                          mixture_a, out_mixture_a, nodes_a, flags_a, node_attributes_a,
//                                                          n, n_mixtures, n_internal_nodes, n_nodes,
//                                                          config);
//        };
//        gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(mixture), dimGrid, dimBlock, fun);
//    }

    return ForwardOutput{out_mixture};
}


} // namespace bvh_mhem_fit

