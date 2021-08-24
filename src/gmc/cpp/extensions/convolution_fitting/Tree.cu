#include "Tree.h"
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
#include "lbvh/building.h"
#include "lbvh/morton_code.h"
#include "util/glm.h"
#include "util/scalar.h"
#include "util/algorithms.h"
#include "util/containers.h"
#include "util/cuda.h"
#include "util/gaussian.h"
#include "util/gaussian_mixture.h"
#include "util/helper.h"
#include "parallel_start.h"

template<typename scalar_t, unsigned N_DIMS>
convolution_fitting::Tree<scalar_t, N_DIMS>::Tree(const at::Tensor* data, const at::Tensor* kernels, const Config& config) : m_config(config), m_data(data), m_kernels(kernels) {
    n = gpe::get_ns(*data);
    kernel_n = gpe::get_ns(*kernels);
    n_channels_in = n.layers;
    n_channels_out = kernel_n.batch;
    n_target_components = n.components * n_channels_in * kernel_n.components;
    TORCH_CHECK(n_channels_in <= (1 << constants::n_bits_for_channel_in_id), "this opperation supports at most " + std::to_string((1 << constants::n_bits_for_channel_in_id)) + " input feature maps")
    TORCH_CHECK(n.components <= (1 << constants::n_bits_for_data_id), "this operation supports at most " + std::to_string((1 << constants::n_bits_for_data_id)) + " data Gaussians")
    TORCH_CHECK(kernel_n.components <= (1 << constants::n_bits_for_kernel_id), "this operation supports at most " + std::to_string((1 << constants::n_bits_for_kernel_id)) + " kernel Gaussians")
    TORCH_CHECK(n_target_components <= std::numeric_limits<index_type>::max(), "Number of components after convolution (target) must be smaller than the largest storable index.")
    TORCH_CHECK(n.batch * n_channels_out < 65535, "n_batch x n_layers must be smaller than 65535 for CUDA")
    TORCH_CHECK(n.components >= 1, "number of components must be greater 1 for this implementation")
    TORCH_CHECK(kernel_n.components >= 1, "number of components must be greater 1 for this implementation")
    TORCH_CHECK(n_channels_in == kernel_n.layers, "number of input feature maps must agree with the second kernel dimension")
    TORCH_CHECK(n.dims == kernel_n.dims, "number of dimensions of data and kernel must agree")
    TORCH_CHECK(n.dims == N_DIMS, "something wrong with dispatch")
    TORCH_CHECK(data->dtype() == kernels->dtype(), "kernel and data dtypes must agree")
    TORCH_CHECK(data->dtype() == caffe2::TypeMeta::Make<scalar_t>(), "something wrong with dispatch, or maybe this float type is not supported.")
    TORCH_CHECK(data->device() == kernels->device(), "data and kernel devices must agree")


    n_leaf_nodes = n_target_components;
    n_internal_nodes = n_leaf_nodes - 1;
    n_nodes = n_leaf_nodes + n_internal_nodes;

}


template<typename scalar_t, unsigned N_DIMS>
at::Tensor convolution_fitting::Tree<scalar_t, N_DIMS>::tree_nodes() const
{
    const auto morton_codes = compute_morton_codes(*m_data, *m_kernels);
    return create_tree_nodes(morton_codes);
}

template<typename scalar_t, unsigned N_DIMS>
at::Tensor convolution_fitting::Tree<scalar_t, N_DIMS>::aabb_from_positions(const at::Tensor& data_positions, const at::Tensor& kernel_positions) const {
    using namespace torch::indexing;
    //    gpe::aabb_from_positions(gpe::positions(data.view({n.batch, 1, n_channels_in * n.components, -1}))) + gpe::aabb_from_positions(gpe::positions(kernels.view({1, n_channels_out, n_channels_in * kernel_n.components, -1})));
    const auto data_upper = std::get<0>(data_positions.max(-2));
    const auto data_lower = std::get<0>(data_positions.min(-2));
    const auto kernel_upper = std::get<0>(kernel_positions.max(-2));
    const auto kernel_lower = std::get<0>(kernel_positions.min(-2));
    const auto upper = data_upper.view({n.batch, 1, n_channels_in, -1}) + kernel_upper.view({1, n_channels_out, n_channels_in, -1});
    const auto lower = data_lower.view({n.batch, 1, n_channels_in, -1}) + kernel_lower.view({1, n_channels_out, n_channels_in, -1});
    const auto upper_upper = std::get<0>(upper.max(2));
    const auto lower_lower = std::get<0>(lower.min(2));
    const auto zeroes = torch::zeros({upper_upper.size(0), upper_upper.size(1), 4 - upper_upper.size(2)}, torch::TensorOptions(upper_upper.device()).dtype(upper_upper.dtype()));
    return torch::cat({upper_upper, zeroes, lower_lower, zeroes}, -1).contiguous();
}

template<typename scalar_t, unsigned N_DIMS>
at::Tensor convolution_fitting::Tree<scalar_t, N_DIMS>::compute_morton_codes(const at::Tensor& data, const at::Tensor& kernels) const {
    const auto data_a = gpe::struct_accessor<typename gpe::Gaussian<N_DIMS, scalar_t>, 3, scalar_t>(data);
    const auto kernel_a = gpe::struct_accessor<typename gpe::Gaussian<N_DIMS, scalar_t>, 3, scalar_t>(kernels);

    auto morton_codes = torch::empty({n.batch, n_channels_out, n_target_components}, torch::TensorOptions(data.device()).dtype(torch::ScalarType::Long));
    auto morton_codes_a = gpe::accessor<uint64_t, 3>(morton_codes);
    const auto aabbs = aabb_from_positions(gpe::positions(data), gpe::positions(kernels));

    assert(aabbs.size(0) == n.batch);
    assert(aabbs.size(1) == n_channels_out);
    assert(aabbs.size(2) == 8);
    auto aabb_a = gpe::accessor<scalar_t, 3>(aabbs);

    dim3 dimBlock = dim3(256, 1, 1);
    dim3 dimGrid = dim3((unsigned(n_target_components) + dimBlock.x - 1) / dimBlock.x,
                        (unsigned(n.batch) + dimBlock.y - 1) / dimBlock.y,
                        (unsigned(n_channels_out) + dimBlock.z - 1) / dimBlock.z);
    gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(data), dimGrid, dimBlock, [morton_codes_a, aabb_a, data_a, kernel_a, *this] __host__ __device__
                                                  (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) mutable {
        GPE_UNUSED(gpe_gridDim)
                // index might not fit into 32 bit, i.e. when n.components == 1 << 17, n_feature_maps_in == 1 << 12 and kernel_n.components == 1 << 4
                // however, such large datasets would be infeasable anyways. i.e., if we have (1<<32) output components, then the morton code array alone takes 8 GB. For one output feature map. For one batch dimension.
                // Sorting alone would probably take too long.
                assert(uint64_t(gpe_blockIdx.x) * uint64_t(gpe_blockDim.x) + uint64_t(gpe_threadIdx.x) <= std::numeric_limits<index_type>::max());
        const index_type component_out_id = gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x;
        if (component_out_id >= n_target_components)
            return;

        const index_type batch_id = gpe_blockIdx.y * gpe_blockDim.y + gpe_threadIdx.y;
        const index_type channel_out_id = gpe_blockIdx.z * gpe_blockDim.z + gpe_threadIdx.z;

        const auto gaussian_indices = gpe::split_n_dim_index<index_type, 3, unsigned>({unsigned(n.components), unsigned(n_channels_in), unsigned(kernel_n.components)}, component_out_id);
        const index_type& component_in_id = gaussian_indices[0];
        const index_type& channel_in_id = gaussian_indices[1];
        const index_type& component_kernel_id = gaussian_indices[2];

        assert(batch_id < n.batch);
        assert(channel_in_id < n_channels_in);
        assert(channel_out_id < n_channels_out);
        assert(component_in_id < n.components);
        assert(component_out_id < n_target_components);
        assert(component_kernel_id < kernel_n.components);

        const auto& data_gaussian = data_a[batch_id][channel_in_id][component_in_id];
        const auto& kernel_gaussian = kernel_a[channel_out_id][channel_in_id][component_kernel_id];
        const auto& aabb_max = gpe::vec<N_DIMS>(aabb_a[batch_id][channel_out_id][0]);
        const auto& aabb_min = gpe::vec<N_DIMS>(aabb_a[batch_id][channel_out_id][4]);
        const auto position = (data_gaussian.position + kernel_gaussian.position - aabb_min) / (aabb_max - aabb_min);
        uint64_t sign = (data_gaussian.weight * kernel_gaussian.weight) > 0;
        uint64_t morton_code = uint64_t(lbvh::morton_code(position));
        // safety check for overlaps
        assert(((sign << 63) & (morton_code << 32)) == 0);
        assert(((morton_code << 32) & uint64_t(component_out_id)) == 0);
        assert(((sign << 63) & uint64_t(component_out_id)) == 0);
        morton_codes_a[batch_id][channel_out_id][component_out_id] = (sign << 63) | (morton_code << 32) | uint64_t(component_out_id);
    });
    return lbvh::sort_morton_codes<uint64_t, int64_t>(morton_codes);
}

template<typename scalar_t, unsigned N_DIMS>
at::Tensor convolution_fitting::Tree<scalar_t, N_DIMS>::create_tree_nodes(const at::Tensor& morton_codes) const {
    using namespace torch::indexing;
    auto n_mixtures = unsigned(n.batch) * n_channels_out;

    auto nodes = torch::ones({n.batch, n_channels_out, n_nodes, 4}, torch::TensorOptions(morton_codes.device()).dtype(gpe::TorchTypeMapper<index_type>::id())) * -1;
    const auto morton_codes_view = morton_codes.view({n_mixtures, n_leaf_nodes});
    const auto morton_codes_a = gpe::accessor<uint64_t, 2>(morton_codes_view);


    { // leaf nodes
        auto nodes_view = nodes.index({Ellipsis, Slice(n_internal_nodes, None), Slice()}).view({n_mixtures, n_leaf_nodes, -1});
        auto nodes_a = gpe::struct_accessor<Node, 2>(nodes_view);
        dim3 dimBlock = dim3(1, 128, 1);
        dim3 dimGrid = dim3((unsigned(n_mixtures) + dimBlock.x - 1) / dimBlock.x,
                            (unsigned(n_leaf_nodes) + dimBlock.y - 1) / dimBlock.y);

        auto fun = [morton_codes_a, nodes_a, n_mixtures, *this] __host__ __device__
                (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) mutable {
            GPE_UNUSED(gpe_gridDim)

                    const auto mixture_id = int(gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x);
            const auto component_id = int(gpe_blockIdx.y * gpe_blockDim.y + gpe_threadIdx.y);
            if (mixture_id >= n_mixtures || component_id >= n_leaf_nodes)
                return;

            const auto& morton_code = morton_codes_a[mixture_id][component_id];
            auto& node = nodes_a[mixture_id][component_id];
            node.object_idx = uint32_t(morton_code); // imo the cast will cut away the morton code. no need for "& 0xfffffff" // uint32_t(morton_code & 0xffffffff);
        };
        gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(morton_codes), dimGrid, dimBlock, fun);
    }
    { // internal nodes
        auto nodes_view = nodes.view({n_mixtures, n_nodes, -1});
        auto nodes_a = gpe::struct_accessor<Node, 2>(nodes_view);
        dim3 dimBlock = dim3(1, 128, 1);
        dim3 dimGrid = dim3((unsigned(n_mixtures) + dimBlock.x - 1) / dimBlock.x,
                            (unsigned(n_internal_nodes) + dimBlock.y - 1) / dimBlock.y);
        auto fun = [morton_codes_a, nodes_a, n_mixtures, *this] __host__ __device__
                (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) mutable {
            GPE_UNUSED(gpe_gridDim)

                    const auto mixture_id = int(gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x);
            const auto node_id = int(gpe_blockIdx.y * gpe_blockDim.y + gpe_threadIdx.y);
            if (mixture_id >= n_mixtures || node_id >= n_internal_nodes)
                return;

            const auto& morton_code = morton_codes_a[mixture_id][0];
            auto& node = nodes_a[mixture_id][node_id];
            //                node.object_idx = lbvh::detail::Node::index_type(0xFFFFFFFF); //  internal nodes // original
            node.object_idx = index_type(node_id);

            const uint2 ij  = lbvh::kernels::determine_range(&morton_code, n_leaf_nodes, node_id);
            const auto gamma = lbvh::kernels::find_split(&morton_code, n_leaf_nodes, ij.x, ij.y);

            node.left_idx  = index_type(gamma);
            node.right_idx = index_type(gamma + 1);
            if(gpe::min(ij.x, ij.y) == gamma)
            {
                node.left_idx += n_leaf_nodes - 1;
            }
            if(gpe::max(ij.x, ij.y) == gamma + 1)
            {
                node.right_idx += n_leaf_nodes - 1;
            }
            assert(node.left_idx != index_type(0xFFFFFFFF));
            assert(node.right_idx != index_type(0xFFFFFFFF));
            nodes_a[mixture_id][int(node.left_idx)].parent_idx = index_type(node_id);
            nodes_a[mixture_id][int(node.right_idx)].parent_idx = index_type(node_id);
        };
        gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(morton_codes), dimGrid, dimBlock, fun);
    }

    return nodes;
}



template class convolution_fitting::Tree<float, 2>;
template class convolution_fitting::Tree<double, 2>;
template class convolution_fitting::Tree<float, 3>;
template class convolution_fitting::Tree<double, 3>;
