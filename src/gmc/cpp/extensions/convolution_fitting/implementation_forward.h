#include "convolution_fitting/implementation.h"
#include <stdio.h>
#include <type_traits>

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include "convolution_fitting/implementation_common.h"
#include "convolution_fitting/Config.h"
#include "common.h"
#include "cuda_qt_creator_definitinos.h"
#include "cuda_operations.h"
#include "hacked_accessor.h"
#include "lbvh/aabb.h"
#include "lbvh/bvh.h"
#include "util/glm.h"
#include "util/scalar.h"
#include "parallel_start.h"
#include "util/algorithms.h"
#include "util/autodiff.h"
#include "util/containers.h"
#include "util/cuda.h"
#include "util/gaussian.h"
#include "util/gaussian_mixture.h"
#include "util/helper.h"
#include "util/mixture.h"


namespace convolution_fitting {



namespace  {

template <typename scalar_t, int N_DIMS>
__host__ __device__
gpe::Gaussian<N_DIMS, scalar_t> convolve(const gpe::Gaussian<N_DIMS, scalar_t>& g1, const gpe::Gaussian<N_DIMS, scalar_t>& g2) {
    constexpr auto a = gcem::pow(scalar_t(2) * glm::pi<scalar_t>(), N_DIMS * scalar_t(0.5));
    const auto b = gpe::sqrt(glm::determinant(g1.covariance) * glm::determinant(g2.covariance));
    gpe::Gaussian<N_DIMS, scalar_t> ret {g1.weight * g2.weight * a * b, g1.position + g2.position, g1.covariance + g2.covariance};
    ret.weight /= gpe::sqrt(glm::determinant(ret.covariance));
    return ret;
}


} // anonymous namespace


template<int REDUCTION_N = 4, typename scalar_t, unsigned N_DIMS>
ForwardOutput forward_impl_t(const torch::Tensor& data, const torch::Tensor& kernels, const Config& config) {
    using namespace torch::indexing;

    const auto n = gpe::get_ns(data);
    const auto kernel_n = gpe::get_ns(kernels);
    const auto n_channels_in = n.layers;
    const auto n_channels_out = kernel_n.batch;
    const auto n_target_components = n.components * n_channels_in * kernel_n.components;
    TORCH_CHECK(n_channels_in <= (1 << constants::n_bits_for_channel_in_id), "this opperation supports at most " + std::to_string((1 << constants::n_bits_for_channel_in_id)) + " input feature maps")
    TORCH_CHECK(n.components <= (1 << constants::n_bits_for_data_id), "this operation supports at most " + std::to_string((1 << constants::n_bits_for_data_id)) + " data Gaussians")
    TORCH_CHECK(kernel_n.components <= (1 << constants::n_bits_for_kernel_id), "this operation supports at most " + std::to_string((1 << constants::n_bits_for_kernel_id)) + " kernel Gaussians")
    TORCH_CHECK(n.batch * n_channels_out < 65535, "n_batch x n_layers must be smaller than 65535 for CUDA")
    TORCH_CHECK(n.components >= 1, "number of components must be greater 1 for this implementation")
    TORCH_CHECK(kernel_n.components >= 1, "number of components must be greater 1 for this implementation")
    TORCH_CHECK(n_channels_in == kernel_n.layers, "number of input feature maps must agree with the second kernel dimension")
    TORCH_CHECK(n.dims == kernel_n.dims, "number of dimensions of data and kernel must agree")
    TORCH_CHECK(n.dims == N_DIMS, "something wrong with dispatch")
    TORCH_CHECK(data.dtype() == kernels.dtype(), "kernel and data dtypes must agree")
    TORCH_CHECK(data.dtype() == caffe2::TypeMeta::Make<scalar_t>(), "something wrong with dispatch, or maybe this float type is not supported.")
    TORCH_CHECK(data.device() == kernels.device(), "data and kernel devices must agree")

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

    auto morton_code_data = torch::empty({n_channels_out, n.batch, n_target_components}, torch::TensorOptions(data.device()).dtype(torch::ScalarType::Long));

    const auto data_a = gpe::struct_accessor<typename gpe::Gaussian<N_DIMS, scalar_t>, 3, scalar_t>(data);
    const auto kernel_a = gpe::struct_accessor<typename gpe::Gaussian<N_DIMS, scalar_t>, 3, scalar_t>(kernels);
    auto morton_codes_a = gpe::accessor<int64_t, 3>(morton_code_data);

    auto out_mixture = torch::empty({n.batch, n_channels_out, n_target_components, data.size(-1)}, torch::TensorOptions(data.device()).dtype(data.dtype()));
    auto out_mixture_a = gpe::struct_accessor<gpe::Gaussian<N_DIMS, scalar_t>, 3, scalar_t>(out_mixture);

    std::cout << "n_target_components: " << n_target_components << std::endl;
    std::cout << "n.batch: " << n.batch << std::endl;
    std::cout << "n_channels_out: " << n_channels_out << std::endl;
    std::cout << "n_channels_in: " << n_channels_in << std::endl;
    std::cout << "kernel_n.components: " << kernel_n.components << std::endl;
    std::cout << "n.components: " << n.components << std::endl;


    dim3 dimBlock = dim3(256, 1, 1);
    dim3 dimGrid = dim3((unsigned(n_target_components) + dimBlock.x - 1) / dimBlock.x,
                        (unsigned(n.batch) + dimBlock.y - 1) / dimBlock.y,
                        (unsigned(n_channels_out) + dimBlock.z - 1) / dimBlock.z);
    gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(data), dimGrid, dimBlock, [morton_codes_a, data_a, kernel_a, out_mixture_a, n_channels_in, n_channels_out, kernel_n, n, n_target_components] __host__ __device__
                                                  (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) mutable {

        // index might not fit into 32 bit, i.e. when n.components == 1 << 17, n_feature_maps_in == 1 << 12 and kernel_n.components == 1 << 4
        // however, such large datasets would be infeasable anyways. i.e., if we have (1<<32) output components, then the morton code array alone takes 8 GB. For one output feature map. For one batch dimension.
        // Sorting alone would probably take too long.
        assert(uint64_t(gpe_blockIdx.x) * uint64_t(gpe_blockDim.x) + uint64_t(gpe_threadIdx.x) < (1ull << 32));
        const unsigned component_out_id = gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x;
        if (component_out_id >= n_target_components)
            return;

        const unsigned batch_id = gpe_blockIdx.y * gpe_blockDim.y + gpe_threadIdx.y;
        const unsigned channel_out_id = gpe_blockIdx.z * gpe_blockDim.z + gpe_threadIdx.z;

        const auto gaussian_indices = gpe::split_n_dim_index<uint32_t, 3, unsigned>({unsigned(n.components), unsigned(n_channels_in), unsigned(kernel_n.components)}, component_out_id);
        const unsigned& component_in_id = gaussian_indices[0];
        const unsigned& channel_in_id = gaussian_indices[1];
        const unsigned& component_kernel_id = gaussian_indices[2];

        assert(batch_id < n.batch);
        assert(channel_in_id < n_channels_in);
        assert(channel_out_id < n_channels_out);
        assert(component_in_id < n.components);
        assert(component_out_id < n_target_components);
        assert(component_kernel_id < kernel_n.components);

        const auto& data_gaussian = data_a[batch_id][channel_in_id][component_in_id];
        const auto& kernel_gaussian = kernel_a[channel_out_id][channel_in_id][component_kernel_id];

        out_mixture_a[int(batch_id)][int(channel_out_id)][int(component_out_id)] = convolve(data_gaussian, kernel_gaussian);

        // todo (get something running quickly):
        // - implement convolution and store in large (not fitted) target array
        // - make a unit test (compare perf and error with python implementation)
        //
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
        // - gradient for convolution alone
        // - gradient for fitting alone (tests only)
        // - merge and test. [we have no trickle down now, yey.]



                                      //                GPE_UNUSED(gpe_gridDim)

                                      //                const auto component_id = int(gpe_blockIdx.y * gpe_blockDim.y + gpe_threadIdx.y);
                                      //                if (mixture_id >= n_mixtures || component_id >= n_components)
                                      //                    return;

                                      //                const gpe::Gaussian<N_DIMS, scalar_t>& gaussian = gpe::gaussian<N_DIMS>(mixture_a[mixture_id][component_id]);
                                      //                const glm::vec<N_DIMS, scalar_t>& cov_max = gpe::vec<N_DIMS>(cov_max_a[mixture_id][0]);
                                      //                const auto cov_diag_nd = gpe::diagonal(gaussian.covariance) / cov_max;
                                      //                const auto cov_diag_3d = make_vec3(cov_diag_nd);

                                      //                const auto& whole = reinterpret_cast<const Aabb<scalar_t>&>(aabb_whole_a[mixture_id][0]);
                                      //                auto& morton_code = reinterpret_cast<morton_cuda_t&>(morton_codes_a[mixture_id][component_id]);

                                      //                auto p = make_glmvec3_of(gaussian.position);
                                      //                p.x -= whole.lower.x;
                                      //                p.y -= whole.lower.y;
                                      //                p.z -= whole.lower.z;
                                      //                p.x /= (whole.upper.x - whole.lower.x);
                                      //                p.y /= (whole.upper.y - whole.lower.y);
                                      //                p.z /= (whole.upper.z - whole.lower.z);


                                      //                assert(component_id < 65535);
                                      //                morton_code = lbvh::morton_code<MORTON_CODE_ALGORITHM>(uint16_t(component_id), p, cov_diag_3d);

                                      //                morton_code = lbvh::morton_code(p);
                                      //                morton_code <<= 32;
                                      //                morton_code |= morton_cuda_t(component_id);
    });

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

