#include "Tree.h"
#include <stdio.h>
#include <type_traits>

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include "convolution_fitting/Config.h"
#include "convolution_fitting/morton_code.h"
#include "convolution_fitting/building.h"
#include "common.h"
#include "cuda_operations.h"
#include "hacked_accessor.h"
#include "util/glm.h"
#include "util/containers.h"
#include "util/cuda.h"
#include "util/helper.h"
#include "parallel_start.h"

template<typename scalar_t, unsigned N_DIMS>
convolution_fitting::Tree<scalar_t, N_DIMS>::Tree(const at::Tensor& data, const at::Tensor& kernels, Data* storage, const Config& config) : m_config(config), m_data(storage) {
    using namespace torch::indexing;
    n = gpe::get_ns(data);
    kernel_n = gpe::get_ns(kernels);
    n_channels_in = index_type(n.layers);
    n_channels_out = index_type(kernel_n.batch);
    n_target_components = n_channels_in * index_type(n.components * kernel_n.components);
    TORCH_CHECK(n_target_components <= std::numeric_limits<index_type>::max(), "this opperation supports at most " + std::to_string(std::numeric_limits<index_type>::max()) + " target components (input channels x input components x kernel components) = (" +
                std::to_string(n_channels_in) + " x " + std::to_string(n.components) + " x " + std::to_string(kernel_n.components) + ")!")

    TORCH_CHECK(index_type(n.batch) * n_channels_out < 65535, "n_batch x n_layers must be smaller than 65535 for CUDA")
    TORCH_CHECK(n.components >= 1, "number of components must be greater 1 for this implementation")
    TORCH_CHECK(kernel_n.components >= 1, "number of components must be greater 1 for this implementation")
    TORCH_CHECK(n_channels_in == index_type(kernel_n.layers), "number of input feature maps must agree with the second kernel dimension")
    TORCH_CHECK(n.dims == kernel_n.dims, "number of dimensions of data and kernel must agree")
    TORCH_CHECK(n.dims == N_DIMS, "something wrong with dispatch")
    TORCH_CHECK(data.dtype() == kernels.dtype(), "kernel and data dtypes must agree")
    TORCH_CHECK(data.dtype() == caffe2::TypeMeta::Make<scalar_t>(), "something wrong with dispatch, or maybe this float type is not supported.")
    TORCH_CHECK(data.device() == kernels.device(), "data and kernel devices must agree")


    n_leaf_nodes = n_target_components;
    n_internal_nodes = n_leaf_nodes - 1;
    n_nodes = n_leaf_nodes + n_internal_nodes;


    m_data->data_weights = gpe::weights(data).contiguous();
    m_data->data_positions = gpe::positions(data).contiguous();
    m_data->data_covariances = data.index({Ellipsis, Slice(N_DIMS + 1, None)}).contiguous();
    m_data->kernel_weights = gpe::weights(kernels).contiguous();
    m_data->kernel_positions = gpe::positions(kernels).contiguous();
    m_data->kernel_covariances = kernels.index({Ellipsis, Slice(N_DIMS + 1, None)}).contiguous();

    data_weights_a = gpe::accessor<scalar_t, 3>(m_data->data_weights);
    data_positions_a = gpe::struct_accessor<Vec, 3>(m_data->data_positions);
    data_covariances_a = gpe::struct_accessor<Mat, 3>(m_data->data_covariances);
    kernel_weights_a = gpe::accessor<scalar_t, 3>(m_data->kernel_weights);
    kernel_positions_a = gpe::struct_accessor<Vec, 3>(m_data->kernel_positions);
    kernel_covariances_a = gpe::struct_accessor<Mat, 3>(m_data->kernel_covariances);
}

template<typename scalar_t, unsigned N_DIMS>
torch::Tensor convolution_fitting::Tree<scalar_t, N_DIMS>::aabb_from_positions(const at::Tensor& data_positions, const at::Tensor& kernel_positions) const {
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
torch::Tensor convolution_fitting::Tree<scalar_t, N_DIMS>::compute_morton_codes() const {
    auto morton_codes = torch::empty({n.batch, n_channels_out, n_target_components}, torch::TensorOptions(device()).dtype(torch::ScalarType::Long));
    auto morton_codes_a = gpe::accessor<uint64_t, 3>(morton_codes);
    const auto aabbs = aabb_from_positions(m_data->data_positions, m_data->kernel_positions);

    assert(aabbs.size(0) == n.batch);
    assert(aabbs.size(1) == n_channels_out);
    assert(aabbs.size(2) == 8);
    auto aabb_a = gpe::accessor<scalar_t, 3>(aabbs);

    dim3 dimBlock = dim3(256, 1, 1);
    dim3 dimGrid = dim3((unsigned(n_target_components) + dimBlock.x - 1) / dimBlock.x,
                        (unsigned(n.batch) + dimBlock.y - 1) / dimBlock.y,
                        (unsigned(n_channels_out) + dimBlock.z - 1) / dimBlock.z);
    gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(morton_codes), dimGrid, dimBlock, [morton_codes_a, aabb_a, *this] __host__ __device__
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

        const auto& data_position = data_positions_a[batch_id][channel_in_id][component_in_id];
        const auto& kernel_position = kernel_positions_a[channel_out_id][channel_in_id][component_kernel_id];
        const auto& aabb_max = gpe::vec<N_DIMS>(aabb_a[batch_id][channel_out_id][0]);
        const auto& aabb_min = gpe::vec<N_DIMS>(aabb_a[batch_id][channel_out_id][4]);
        const auto position = (data_position + kernel_position - aabb_min) / (aabb_max - aabb_min);

        const auto& data_weight = data_weights_a[batch_id][channel_in_id][component_in_id];
        const auto& kernel_weight = kernel_weights_a[channel_out_id][channel_in_id][component_kernel_id];
        uint64_t sign = (data_weight * kernel_weight) > 0;
        uint64_t morton_code = uint64_t(lbvh::morton_code(position));
        // safety check for overlaps
        assert(((sign << 62) & (morton_code << 32)) == 0);
        assert(((morton_code << 32) & uint64_t(component_out_id)) == 0);
        assert(((sign << 62) & uint64_t(component_out_id)) == 0);
        morton_codes_a[batch_id][channel_out_id][component_out_id] = (sign << 62) | (morton_code << 32) | uint64_t(component_out_id);
    });
    return lbvh::sort_morton_codes<uint64_t, int64_t>(morton_codes);
}

template<typename scalar_t, unsigned N_DIMS>
void convolution_fitting::Tree<scalar_t, N_DIMS>::create_tree_nodes() {
    using namespace torch::indexing;
    auto n_mixtures = unsigned(n.batch) * n_channels_out;
    const at::Tensor morton_codes = compute_morton_codes();

    auto nodes = torch::ones({n.batch, n_channels_out, n_nodes, 4}, torch::TensorOptions(morton_codes.device()).dtype(gpe::TorchTypeMapper<index_type>::id())) * -1;
    auto nodesobjs = torch::ones({n.batch, n_channels_out, n_nodes}, torch::TensorOptions(morton_codes.device()).dtype(gpe::TorchTypeMapper<index_type>::id()));

    const auto morton_codes_view = morton_codes.view({n_mixtures, n_leaf_nodes});
    const auto morton_codes_a = gpe::accessor<uint64_t, 2>(morton_codes_view);


    { // leaf nodes
        auto nodes_view = nodes.index({Ellipsis, Slice(n_internal_nodes, None), Slice()}).view({n_mixtures, n_leaf_nodes, -1});
        auto nodesobjs_view = nodesobjs.index({Ellipsis, Slice(n_internal_nodes, None)}).view({n_mixtures, n_leaf_nodes});

        auto nodes_a = gpe::struct_accessor<Node, 2>(nodes_view);
        auto nodesobjs_a = gpe::accessor<index_type, 2>(nodesobjs_view);

        dim3 dimBlock = dim3(1, 128, 1);
        dim3 dimGrid = dim3((unsigned(n_mixtures) + dimBlock.x - 1) / dimBlock.x,
                            (unsigned(n_leaf_nodes) + dimBlock.y - 1) / dimBlock.y);

        auto fun = [morton_codes_a, nodes_a, nodesobjs_a, n_mixtures, *this] __host__ __device__
                (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) mutable {
            GPE_UNUSED(gpe_gridDim)

            const auto mixture_id = gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x;
            const auto component_id = gpe_blockIdx.y * gpe_blockDim.y + gpe_threadIdx.y;
            if (mixture_id >= n_mixtures || component_id >= n_leaf_nodes)
                return;

            const auto& morton_code = morton_codes_a[mixture_id][component_id];
            auto& node = nodes_a[mixture_id][component_id];
            node.object_idx = uint32_t(morton_code); // imo the cast will cut away the morton code. no need for "& 0xfffffff" // uint32_t(morton_code & 0xffffffff);

            nodesobjs_a[mixture_id][component_id] = uint32_t(morton_code);
        };
        gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(morton_codes), dimGrid, dimBlock, fun);
    }
    { // internal nodes
        auto nodes_view = nodes.view({n_mixtures, n_nodes, -1});
        auto nodesobjs_view = nodesobjs.view({n_mixtures, n_nodes});

        auto nodes_a = gpe::struct_accessor<Node, 2>(nodes_view);
        auto nodesobjs_a = gpe::accessor<index_type, 2>(nodesobjs_view);

        dim3 dimBlock = dim3(1, 128, 1);
        dim3 dimGrid = dim3((unsigned(n_mixtures) + dimBlock.x - 1) / dimBlock.x,
                            (unsigned(n_internal_nodes) + dimBlock.y - 1) / dimBlock.y);
        auto fun = [morton_codes_a, nodes_a, nodesobjs_a, n_mixtures, *this] __host__ __device__
                (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) mutable {
            GPE_UNUSED(gpe_gridDim)

            const auto mixture_id = gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x;
            const auto node_id = gpe_blockIdx.y * gpe_blockDim.y + gpe_threadIdx.y;
            if (mixture_id >= n_mixtures || node_id >= n_internal_nodes)
                return;

            const auto& morton_code = morton_codes_a[mixture_id][0];
            auto& node = nodes_a[mixture_id][node_id];
            //                node.object_idx = lbvh::detail::Node::index_type(0xFFFFFFFF); //  internal nodes // original
//            node.object_idx = index_type(node_id);
//            nodesobjs_a[mixture_id][node_id] = index_type(node_id);

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
            nodes_a[mixture_id][node.left_idx].parent_idx = index_type(node_id);
            nodes_a[mixture_id][node.right_idx].parent_idx = index_type(node_id);
        };
        gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(morton_codes), dimGrid, dimBlock, fun);
    }

    m_data->nodes = nodes;
    nodes_a = gpe::struct_accessor<typename Tree::Node, 3>(m_data->nodes);

    m_data->nodesobjs = nodesobjs;
    nodesobjs_a = gpe::accessor<index_type, 3>(m_data->nodesobjs);
}

template<typename scalar_t, unsigned N_DIMS>
void convolution_fitting::Tree<scalar_t, N_DIMS>::create_attributes()
{
    m_data->node_attributes = torch::zeros({n.batch, n_channels_out, n_internal_nodes, sizeof (typename Tree::NodeAttributes)}, torch::TensorOptions(device()).dtype(torch::ScalarType::Byte));
    node_attributes_a = gpe::struct_accessor<typename Tree::NodeAttributes, 3>(m_data->node_attributes);

    dim3 dimBlock = dim3(256, 1, 1);
    dim3 dimGrid = dim3((unsigned(n_target_components) + dimBlock.x - 1) / dimBlock.x,
                        (unsigned(n_channels_out) + dimBlock.y - 1) / dimBlock.y,
                        (unsigned(n.batch) + dimBlock.z - 1) / dimBlock.z);

    gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(m_data->nodes), dimGrid, dimBlock, [*this] __host__ __device__
                                                  (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) mutable {
        // index might not fit into 32 bit, i.e. when n.components == 1 << 17, n_feature_maps_in == 1 << 12 and kernel_n.components == 1 << 4
        // however, such large datasets would be infeasable anyways. i.e., if we have (1<<32) output components, then the morton code array alone takes 8 GB. For one output feature map. For one batch dimension.
        // Sorting alone would probably take too long.
        assert(uint64_t(gpe_blockIdx.x) * uint64_t(gpe_blockDim.x) + uint64_t(gpe_threadIdx.x) < (1ull << 32));
        const unsigned component_out_id = gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x;
        if (component_out_id >= n_target_components)
            return;

//        printf("component_out_id: %d\n", component_out_id);
        const unsigned channel_out_id = gpe_blockIdx.y * gpe_blockDim.y + gpe_threadIdx.y;
        const unsigned batch_id = gpe_blockIdx.z * gpe_blockDim.z + gpe_threadIdx.z;
//        const typename Tree::Node& leaf_node = nodes_a[batch_id][channel_out_id][tree.n_internal_nodes + component_out_id];

        const auto get_node = [&](typename Tree::index_type node_id) -> const auto& {
            assert(node_id < n_nodes);
            return nodes_a[batch_id][channel_out_id][node_id];
        };
        const auto get_attribs = [&](typename Tree::index_type node_id) -> auto& {
            assert(node_id < n_internal_nodes);
            return node_attributes_a[batch_id][channel_out_id][node_id];
        };

        const Node& leaf_node = get_node(n_internal_nodes + component_out_id);

        const auto gaussian_indices = gpe::split_n_dim_index<uint32_t, 3, unsigned>({unsigned(n.components), unsigned(n_channels_in), unsigned(kernel_n.components)}, leaf_node.object_idx);
        const unsigned& component_in_id = gaussian_indices[0];
        const unsigned& channel_in_id = gaussian_indices[1];
        const unsigned& component_kernel_id = gaussian_indices[2];

        assert(batch_id < n.batch);
        assert(channel_in_id < n_channels_in);
        assert(channel_out_id < n_channels_out);
        assert(component_in_id < n.components);
        assert(component_out_id < n_target_components);
        assert(component_kernel_id < kernel_n.components);

        const auto& data_weight = data_weights_a[batch_id][channel_in_id][component_in_id];
        const auto& kernel_weight = kernel_weights_a[channel_out_id][channel_in_id][component_kernel_id];

        auto mass = data_weight * kernel_weight;        // the integral of w * N(..) is w  since N() integrates to 1; the convolution of two weighted Gaussian distribution has a weight w1 * w2.
        auto count = typename Tree::index_type(1);

        auto node_id = leaf_node.parent_idx;
        while (node_id < n_internal_nodes) {
            const auto& node = get_node(node_id);
            auto& attribs = get_attribs(node_id);
            mass += gpe::atomicAdd(&attribs.mass, mass);
            auto count_tmp = gpe::atomicAdd(&attribs.n_gaussians, count);
            if (count_tmp == 0)
                break;
            count += count_tmp;
            node_id = node.parent_idx;
        }
    });
}

template<typename scalar_t, unsigned N_DIMS>
void convolution_fitting::Tree<scalar_t, N_DIMS>::select_fitting_subtrees()
{
    const auto n_mixtures = unsigned(n.batch) * n_channels_out;
    m_data->fitting_subtrees = torch::ones({n.batch, n_channels_out, 2 * m_config.n_components_fitting}, torch::TensorOptions(device()).dtype(gpe::TorchTypeMapper<typename Tree::index_type>::id())) * -1;
    fitting_subtrees_a = gpe::accessor<typename Tree::index_type, 3>(m_data->fitting_subtrees);

#ifdef GPE_SORT_FITTED
    auto fitlengths = torch::zeros({n.batch, n_channels_out, m_config.n_components_fitting},  torch::TensorOptions(device()).dtype(gpe::TorchTypeMapper<typename Tree::index_type>::id()));
    auto fitlengths_a = gpe::accessor<typename Tree::index_type, 3>(fitlengths);
#endif

    auto selected_nodes_rating = torch::empty({n.batch, n_channels_out, m_config.n_components_fitting}, torch::TensorOptions(device()).dtype(gpe::TorchTypeMapper<scalar_t>::id()));
    auto selected_nodes_rating_a = gpe::accessor<scalar_t, 3>(selected_nodes_rating);

    dim3 dimBlock = dim3(32, 1, 1);
    dim3 dimGrid = dim3((uint(n_mixtures) + dimBlock.x - 1) / dimBlock.x, 1, 1);
    gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(m_data->nodes), dimGrid, dimBlock, [=, *this] __host__ __device__
                                                  (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) mutable {
        GPE_UNUSED(gpe_gridDim)

        const auto mixture_id = gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x;
        if (mixture_id >= n_mixtures)
            return;

        const auto mixture_indices = gpe::split_n_dim_index<uint32_t, 3, unsigned>({unsigned(n.batch), unsigned(n_channels_out)}, mixture_id);
        const auto& batch_id = mixture_indices[0];
        const auto& channel_out_id = mixture_indices[1];

        const auto get_node = [&](index_type node_id) -> const auto& {
            assert(node_id < n_nodes);
            return nodes_a[batch_id][channel_out_id][node_id];
        };
        const auto get_attribs = [&](index_type node_id) -> auto& {
            assert(node_id < n_internal_nodes);
            return node_attributes_a[batch_id][channel_out_id][node_id];
        };

        unsigned n_selected_components = 0;
        auto compute_rating = [&](index_type node_id) -> scalar_t {
            assert(node_id < n_nodes);
            // todo: will break with negative weights, should compute sum of abs integrals / seperately positive and negative integrals
            if (node_id >= n_internal_nodes)
                return -2; // -2 so it's safely below -1 from cach_id_with_highest_rating
            else
                return gpe::abs(get_attribs(node_id).mass);
        };
        auto cach_id_with_highest_rating = [&]() {
            scalar_t rating = -1;
            unsigned best_node_id = unsigned(-1);
            for (unsigned i = 0; i < n_selected_components; ++i) {
                if (selected_nodes_rating_a[batch_id][channel_out_id][i] > rating) {
                    rating = selected_nodes_rating_a[batch_id][channel_out_id][i];
                    best_node_id = i;
                }
            }
            // can become unsigned(-1) when no selectable node remains
            return best_node_id;
        };
        auto write_selected_node = [&](index_type position, index_type node_id) {
            fitting_subtrees_a[batch_id][channel_out_id][2 * position] = node_id;
            selected_nodes_rating_a[batch_id][channel_out_id][position] = compute_rating(node_id);
#ifdef GPE_SORT_FITTED
            fitlengths_a[batch_id][channel_out_id][position] = node_id < n_internal_nodes ? get_attribs(node_id).n_gaussians : 1;
#endif
        };
        write_selected_node(0, 0);
        n_selected_components = 1;

        while (n_selected_components < m_config.n_components_fitting)  {
            auto best_node_cache_id = cach_id_with_highest_rating();
            if (best_node_cache_id >= n_selected_components)
                break;  // ran out of nodes
            auto best_node_id = fitting_subtrees_a[batch_id][channel_out_id][2 * best_node_cache_id];
            assert(best_node_id < n_nodes);
            const auto& best_descend_node = get_node(best_node_id);
            assert(best_node_id < n_internal_nodes); // we should have only internal nodes at this point as cach_id_with_highest_rating() returns 0xffff.. if the node is not full.

            write_selected_node(best_node_cache_id, best_descend_node.left_idx);
            write_selected_node(n_selected_components++, best_descend_node.right_idx);
        }

        auto write_start_end = [&](index_type position) {
            const auto fitting_root_node_id = fitting_subtrees_a[batch_id][channel_out_id][2*position];

            // getting start and end leaf by descending to the leftest and rightest leaf, respectively
            auto start_id = fitting_root_node_id;
            auto current_id = start_id;
            do { // left descend
                start_id = current_id;
                current_id = get_node(current_id).left_idx;
            } while (current_id != index_type(-1));

            auto end_id = fitting_root_node_id;
            current_id = end_id;
            do { // right descend
                end_id = current_id;
                current_id = get_node(current_id).right_idx;
            } while (current_id != index_type(-1));
            ++end_id; // it should point past the back

            fitting_subtrees_a[batch_id][channel_out_id][2 * position] = start_id;
            fitting_subtrees_a[batch_id][channel_out_id][2 * position + 1] = end_id;
        };

        for(int i = 0; i < n_selected_components; i++)
        {
            write_start_end(i);
        }
    });

#ifdef GPE_SORT_FITTED
    const int num_segments = n.batch * n_channels_out;
    const int num_components = m_config.n_components_fitting;
    auto fitlengths_orig = fitlengths.clone();
    const index_type* d_keys_in = reinterpret_cast<const index_type*>(fitlengths_orig.data_ptr());
    index_type* d_keys_out = reinterpret_cast<index_type*>(fitlengths.data_ptr());
    auto fitting_subtrees_orig = m_data->fitting_subtrees.clone();
    const index_type* d_vals_in = reinterpret_cast<const index_type*>(fitting_subtrees_orig.data_ptr());
    index_type* d_vals_out = reinterpret_cast<index_type*>(m_data->fitting_subtrees.data_ptr());

    if (fitlengths.is_cuda())
    {
        int num_items = int(fitlengths.numel());
        const torch::Tensor offsets = torch::arange(0, num_segments + 1, torch::TensorOptions(fitlengths.device()).dtype(torch::ScalarType::Int)) * num_components;
        int* d_offsets = offsets.data_ptr<int>();

        // Determine temporary device storage requirements
        void     *d_temp_storage = nullptr;
        size_t   temp_storage_bytes = 0;

        cub::DeviceSegmentedRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out, d_vals_in, d_vals_out,
        num_items, num_segments,
        d_offsets, d_offsets + 1, 0, 32);

        GPE_CUDA_ASSERT(cudaPeekAtLastError());
        GPE_CUDA_ASSERT(cudaDeviceSynchronize());

        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        GPE_CUDA_ASSERT(cudaPeekAtLastError());
        GPE_CUDA_ASSERT(cudaDeviceSynchronize());
        // Run sorting operation

        cub::DeviceSegmentedRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out, d_vals_in, d_vals_out,
        num_items, num_segments,
        d_offsets, d_offsets + 1, 0, 32);

        // d_keys_out            <-- [6, 8, 5, 7, 0, 3, 8, 9]
        // d_values_out          <-- [1, 0, 3, 2, 5, 4, 7, 6]

        GPE_CUDA_ASSERT(cudaPeekAtLastError());
        GPE_CUDA_ASSERT(cudaDeviceSynchronize());

        cudaFree(d_temp_storage);
    }
    else {
        throw std::runtime_error("No solution yet!");
    }
#endif

}

template<typename scalar_t, unsigned N_DIMS>
void convolution_fitting::Tree<scalar_t, N_DIMS>::set_nodes_and_friends(const at::Tensor& nodes, const at::Tensor& nodesobjs, const at::Tensor& node_attributes, const at::Tensor& fitting_subtrees)
{
    m_data->nodes = nodes;
    m_data->nodesobjs = nodesobjs;
    m_data->node_attributes = node_attributes;
    m_data->fitting_subtrees = fitting_subtrees;

    nodesobjs_a = gpe::accessor<typename Tree::index_type, 3>(m_data->nodesobjs);
    nodes_a = gpe::struct_accessor<typename Tree::Node, 3>(m_data->nodes);
    node_attributes_a = gpe::struct_accessor<typename Tree::NodeAttributes, 3>(m_data->node_attributes);
    fitting_subtrees_a = gpe::accessor<typename Tree::index_type, 3>(m_data->fitting_subtrees);
}

template<typename scalar_t, unsigned N_DIMS>
void convolution_fitting::Tree<scalar_t, N_DIMS>::set_friends(const at::Tensor& nodesobjs, const at::Tensor& fitting_subtrees)
{
    m_data->nodesobjs = nodesobjs;
    m_data->fitting_subtrees = fitting_subtrees;

    nodesobjs_a = gpe::accessor<typename Tree::index_type, 3>(m_data->nodesobjs);
    fitting_subtrees_a = gpe::accessor<typename Tree::index_type, 3>(m_data->fitting_subtrees);
}


template class convolution_fitting::Tree<float, 2>;
template class convolution_fitting::Tree<double, 2>;
template class convolution_fitting::Tree<float, 3>;
template class convolution_fitting::Tree<double, 3>;

