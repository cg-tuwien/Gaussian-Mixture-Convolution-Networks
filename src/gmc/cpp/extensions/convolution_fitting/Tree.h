#ifndef CONVOLUTION_FITTING_TREE_H
#define CONVOLUTION_FITTING_TREE_H

#include <torch/types.h>

#include "util/mixture.h"
#include "convolution_fitting/Config.h"


namespace convolution_fitting {


template<typename scalar_t, unsigned N_DIMS>
class Tree {
public:
    gpe::MixtureNs n;
    gpe::MixtureNs kernel_n;
    using index_type = uint32_t;
    index_type n_channels_in = 0;
    index_type n_channels_out = 0;
    index_type n_target_components;
    index_type n_leaf_nodes = 0;
    index_type n_internal_nodes = 0;
    index_type n_nodes = 0;

    const Config m_config;

    const torch::Tensor* m_data = nullptr;
    const torch::Tensor* m_kernels = nullptr;

    struct Node
    {
        index_type parent_idx; // parent node
        index_type left_idx;   // index of left  child node
        index_type right_idx;  // index of right child node
        index_type object_idx; // == 0xFFFFFFFF if internal node.
    };

    struct NodeAttributes {
        scalar_t mass;
        index_type n_gaussians;
    };


    Tree(const torch::Tensor* data, const torch::Tensor* kernels, const Config& config);
    at::Tensor tree_nodes() const;

    inline torch::Tensor aabb_from_positions(const torch::Tensor& data_positions, const torch::Tensor& kernel_positions) const;
    torch::Tensor compute_morton_codes(const torch::Tensor& data, const torch::Tensor& kernels) const;
    at::Tensor create_tree_nodes(const at::Tensor& morton_codes) const;
};


} // namespace bvh_mhem_fit

#endif // CONVOLUTION_FITTING_TREE_H
