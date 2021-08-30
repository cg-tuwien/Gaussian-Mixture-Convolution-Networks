#include "convolution_fitting/implementation.h"

#include <cuda.h>
#include <torch/types.h>

#include "convolution_fitting/Config.h"
#include "convolution_fitting/Tree.h"
#include "cuda_qt_creator_definitinos.h"
#include "cuda_operations.h"
#include "hacked_accessor.h"
#include "util/gaussian.h"
#include "util/grad/welford.h"
#include "util/helper.h"
#include "util/welford.h"
#include "parallel_start.h"


namespace convolution_fitting {

// this method is already written with the (new) Gaussian PDF formulation!!

template<int REDUCTION_N = 4, typename scalar_t, unsigned N_DIMS>
std::pair<torch::Tensor, torch::Tensor> backward_impl_t(const torch::Tensor& grad, const ForwardOutput& forward_out, const Config& config) {
    using Tree = Tree<scalar_t, N_DIMS>;
    using G = gpe::Gaussian<N_DIMS, scalar_t>;

    TORCH_CHECK(config.n_components_fitting <= N_MAX_TARGET_COMPS, "can't fit more than " + std::to_string(N_MAX_TARGET_COMPS) + " components")

    typename Tree::Data tree_data_storage;
    Tree tree(forward_out.data, forward_out.kernels, &tree_data_storage, config);
    tree.create_tree_nodes();
    tree.create_attributes();
    tree.select_fitting_subtrees();

    torch::Tensor out_mixture = forward_out.fitting;
    auto out_mixture_a = gpe::struct_accessor<G, 3>(out_mixture);
    auto cached_pos_covs = forward_out.cached_pos_covs;
    auto cached_pos_covs_a = gpe::struct_accessor<typename Tree::Mat, 3>(cached_pos_covs);
    auto incoming_grad_a = gpe::struct_accessor<G, 3>(grad);

    auto grad_data = torch::zeros_like(forward_out.data);
    auto grad_kernels = torch::zeros_like(forward_out.kernels);

    auto grad_data_a = gpe::struct_accessor<G, 3>(grad_data);
    auto grad_kernels_a = gpe::struct_accessor<G, 3>(grad_kernels);


    { // fit subtrees
        dim3 dimBlock = dim3(32, 1, 1);
        dim3 dimGrid = dim3((unsigned(config.n_components_fitting) + dimBlock.x - 1) / dimBlock.x,
                            (unsigned(tree.n_channels_out) + dimBlock.y - 1) / dimBlock.y,
                            (unsigned(tree.n.batch) + dimBlock.z - 1) / dimBlock.z);
        gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(out_mixture), dimGrid, dimBlock, [=] __host__ __device__
                                                      (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) mutable {
            GPE_UNUSED(gpe_gridDim)
            using index_type = typename Tree::index_type;

            const unsigned component_out_id = gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x;
            if (component_out_id >= config.n_components_fitting)
                return;

            //        printf("component_out_id: %d\n", component_out_id);
            const unsigned channel_out_id = gpe_blockIdx.y * gpe_blockDim.y + gpe_threadIdx.y;
            const unsigned batch_id = gpe_blockIdx.z * gpe_blockDim.z + gpe_threadIdx.z;
            assert(batch_id < tree.n.batch);
            assert(channel_out_id < tree.n_channels_out);

            const auto fitting_root_node_id = tree.fitting_subtrees_a[batch_id][channel_out_id][component_out_id];
            if (fitting_root_node_id >= config.n_components_fitting) {
                return;
            }

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

            const G& g = out_mixture_a[batch_id][channel_out_id][component_out_id];
            const G& incoming_grad = incoming_grad_a[batch_id][channel_out_id][component_out_id];
            const typename G::cov_t& cached_pos_cov = cached_pos_covs_a[batch_id][channel_out_id][component_out_id];

            gpe::grad::WeightedMeanAndCov<N_DIMS, scalar_t> pos_aggregator(g.weight, g.position, cached_pos_cov,
                                                                           incoming_grad.weight, incoming_grad.position, incoming_grad.covariance);
            gpe::grad::WeightedMean<scalar_t, typename G::cov_t> cov_aggregator(g.weight, g.covariance - cached_pos_cov,
                                                                                incoming_grad.weight, incoming_grad.covariance);
            for (index_type k = start_id; k < end_id; ++k) {
                const auto target_component_id = get_node(k).object_idx;

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

                scalar_t grad_weight1;
                typename G::pos_t grad_pos;
                pos_aggregator.addValue(convolved_weight, convolved_position, &grad_weight1, &grad_pos);

                scalar_t grad_weight2;
                typename G::cov_t grad_cov;
                cov_aggregator.addValue(convolved_weight, convolved_covariance, &grad_weight2, &grad_cov);

                G& grad_data = grad_data_a[batch_id][channel_in_id][component_in_id];
                G& grad_kernel = grad_kernels_a[channel_out_id][channel_in_id][component_kernel_id];
                gpe::atomicAdd(&grad_data.weight, (grad_weight1 + grad_weight2) * kernel_weight);
                gpe::atomicAdd(&grad_kernel.weight, (grad_weight1 + grad_weight2) * data_weight);
                for (unsigned i = 0; i < N_DIMS; ++i) {
                    gpe::atomicAdd(&grad_data.position[i], grad_pos[i]);
                    gpe::atomicAdd(&grad_kernel.position[i], grad_pos[i]);
                    for (unsigned j = 0; j < N_DIMS; ++j) {
                        gpe::atomicAdd(&grad_data.covariance[i][j], grad_cov[i][j]);
                        gpe::atomicAdd(&grad_kernel.covariance[i][j], grad_cov[i][j]);
                    }
                }
            }
        });
    }

    return {grad_data, grad_kernels};
}

} // namespace bvh_mhem_fit

