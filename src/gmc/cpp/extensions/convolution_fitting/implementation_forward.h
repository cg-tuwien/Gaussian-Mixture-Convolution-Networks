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
#include "util/welford.h"
#include "pieces/integrate.h"
#include "parallel_start.h"


namespace convolution_fitting {

template<int REDUCTION_N = 4, typename scalar_t, unsigned N_DIMS>
ForwardOutput forward_impl_t(const torch::Tensor& data, const torch::Tensor& kernels, const Config& config) {
    // cont with fitting todos:
    // - split data into weight, pos and cov tensors.
    // - use pdf formulation to avoid complicated integral and convolution
    // - if 1G bad, implement 2G and 4G fitting.
    //
    // cont with gradient computation
    // - gradient for fitting alone (tests only)
    // - merge and test. [we have no trickle down now, yey.]

    using Tree = Tree<scalar_t, N_DIMS>;

    TORCH_CHECK(config.n_components_fitting <= N_MAX_TARGET_COMPS, "can't fit more than " + std::to_string(N_MAX_TARGET_COMPS) + " components")

    typename Tree::Data tree_data_storage;
    Tree tree(data, kernels, &tree_data_storage, config);
    tree.create_tree_nodes();
    tree.create_attributes();
    tree.select_fitting_subtrees();

    auto out_mixture = torch::empty({tree.n.batch, tree.n_channels_out, config.n_components_fitting, data.size(-1)}, torch::TensorOptions(data.device()).dtype(data.dtype()));
    auto out_mixture_a = gpe::struct_accessor<gpe::Gaussian<N_DIMS, scalar_t>, 3>(out_mixture);


    { // fit subtrees
        dim3 dimBlock = dim3(32, 1, 1);
        dim3 dimGrid = dim3((unsigned(config.n_components_fitting) + dimBlock.x - 1) / dimBlock.x,
                            (unsigned(tree.n_channels_out) + dimBlock.y - 1) / dimBlock.y,
                            (unsigned(tree.n.batch) + dimBlock.z - 1) / dimBlock.z);
        gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(data), dimGrid, dimBlock, [=] __host__ __device__
                                                      (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) mutable {
            GPE_UNUSED(gpe_gridDim)
            using G = gpe::Gaussian<N_DIMS, scalar_t>;
            using index_type = typename Tree::index_type;

            const unsigned component_out_id = gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x;
            if (component_out_id >= config.n_components_fitting)
                return;

            //        printf("component_out_id: %d\n", component_out_id);
            const unsigned channel_out_id = gpe_blockIdx.y * gpe_blockDim.y + gpe_threadIdx.y;
            const unsigned batch_id = gpe_blockIdx.z * gpe_blockDim.z + gpe_threadIdx.z;
            assert(batch_id < tree.n.batch);
            assert(channel_out_id < tree.n_channels_out);

            const auto fitting_root_node_id = tree->fitting_subtrees_a[batch_id][channel_out_id][component_out_id];

            const auto get_node = [&](index_type node_id) -> const typename Tree::Node& {
                assert(node_id < tree.n_nodes);
                return tree.nodes_a[batch_id][channel_out_id][node_id];
            };
            const auto get_attribs = [&](index_type node_id) -> typename Tree::NodeAttributes& {
                assert(node_id < tree.n_internal_nodes);
                return tree.node_attributes_a[batch_id][channel_out_id][node_id];
            };


            // fitting one Gaussian, all target Gaussians are equally important, but posses different weights on their own.

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

            gpe::WeightedMeanAndCov<N_DIMS, scalar_t> pos_aggregator;
            gpe::WeightedMean<scalar_t, typename G::cov_t> cov_aggregator;
            for (index_type i = start_id; i < end_id; ++i) {
                const auto target_component_id = get_node(i).object_idx;

                const auto gaussian_indices = gpe::split_n_dim_index<uint32_t, 3, unsigned>({unsigned(tree.n.components), unsigned(tree.n_channels_in), unsigned(tree.kernel_n.components)}, target_component_id);
                const unsigned& component_in_id = gaussian_indices[0];
                const unsigned& channel_in_id = gaussian_indices[1];
                const unsigned& component_kernel_id = gaussian_indices[2];
                assert(component_in_id < tree.n.components);
                assert(channel_in_id < tree.n_channels_in);
                assert(component_kernel_id < tree.kernel_n.components);

                const auto data_weight = tree.data_weights_a[batch_id][channel_in_id][component_in_id];
                const auto& data_position = tree.data_positions_a[batch_id][channel_in_id][component_in_id];
                const auto& data_covariance = tree.data_covariances_a[batch_id][channel_in_id][component_in_id];

                const auto kernel_weight = tree.kernel_weights_a[channel_out_id][channel_in_id][component_kernel_id];
                const auto& kernel_position = tree.kernel_positions_a[channel_out_id][channel_in_id][component_kernel_id];
                const auto& kernel_covariance = tree.kernel_covariances_a[channel_out_id][channel_in_id][component_kernel_id];

                auto convolved_weight = data_weight * kernel_weight;
                auto convolved_position = data_position + kernel_position;
                auto convolved_covariance = data_covariance + kernel_covariance;

                pos_aggregator.addValue(convolved_weight, convolved_position);
                cov_aggregator.addValue(convolved_weight, convolved_covariance);
            }
            const auto cov_mat = cov_aggregator.mean() + pos_aggregator.cov_matrix();
            const auto g = G{pos_aggregator.w_sum * gpe::gaussian_amplitude(cov_mat), pos_aggregator.mean(), cov_mat};
            out_mixture_a[int(batch_id)][int(channel_out_id)][int(component_out_id)] = g;
        });
    }


    return ForwardOutput{out_mixture};
}


} // namespace bvh_mhem_fit

