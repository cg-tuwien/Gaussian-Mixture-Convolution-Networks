#ifndef CONVOLUTION_FITTING_TREE_H
#define CONVOLUTION_FITTING_TREE_H

#include <torch/types.h>

#include "util/glm.h"
#include "hacked_accessor.h"
#include "util/mixture.h"
#include "convolution_fitting/Config.h"


namespace convolution_fitting {

template<typename scalar_t, unsigned N_DIMS>
class Tree {
public:
    using index_type = uint32_t;
    using Vec = glm::vec<N_DIMS, scalar_t>;
    using Mat = glm::mat<N_DIMS, N_DIMS, scalar_t>;

    struct Data {
        torch::Tensor data_weights;
        torch::Tensor data_positions;
        torch::Tensor data_covariances;
        torch::Tensor kernel_weights;
        torch::Tensor kernel_positions;
        torch::Tensor kernel_covariances;
        torch::Tensor nodes;
        torch::Tensor nodesobjs;
        torch::Tensor node_attributes;
        torch::Tensor fitting_subtrees;
    };

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
    gpe::MixtureNs n;
    gpe::MixtureNs kernel_n;
    index_type n_channels_in = 0;
    index_type n_channels_out = 0;
    index_type n_target_components;
    index_type n_leaf_nodes = 0;
    index_type n_internal_nodes = 0;
    index_type n_nodes = 0;

    const Config m_config;

    gpe::PackedTensorAccessor32<scalar_t, 3> data_weights_a;
    gpe::PackedTensorAccessor32<Vec, 3> data_positions_a;
    gpe::PackedTensorAccessor32<Mat, 3> data_covariances_a;
    gpe::PackedTensorAccessor32<scalar_t, 3> kernel_weights_a;
    gpe::PackedTensorAccessor32<Vec, 3> kernel_positions_a;
    gpe::PackedTensorAccessor32<Mat, 3> kernel_covariances_a;
    gpe::PackedTensorAccessor32<typename Tree::index_type, 3> nodesobjs_a;
    gpe::PackedTensorAccessor32<typename Tree::Node, 3> nodes_a;
    gpe::PackedTensorAccessor32<typename Tree::NodeAttributes, 3> node_attributes_a;
    gpe::PackedTensorAccessor32<typename Tree::index_type, 3> fitting_subtrees_a;
    Data *const m_data = nullptr; // can't store tensors directly, because tree is copied to gpu and Tensors are causing problems when using *this in lambdas


    Tree(const torch::Tensor& data, const torch::Tensor& kernels, Data* storage, const Config& config);
    torch::Device device() const { return m_data->data_weights.device(); }
    caffe2::TypeMeta dtype() const { return m_data->data_weights.dtype(); }
    inline torch::Tensor aabb_from_positions(const torch::Tensor& data_positions, const torch::Tensor& kernel_positions) const;
    torch::Tensor compute_morton_codes() const;
    void create_tree_nodes();
    void create_attributes();
    void select_fitting_subtrees();
    void set_friends(const torch::Tensor& nodesobjs, const torch::Tensor& fitting_subtrees);
    void set_nodes_and_friends(const torch::Tensor& nodes, const torch::Tensor& nodesobjs, const torch::Tensor& node_attributes, const torch::Tensor& fitting_subtrees);
};


} // namespace bvh_mhem_fit

#endif // CONVOLUTION_FITTING_TREE_H
