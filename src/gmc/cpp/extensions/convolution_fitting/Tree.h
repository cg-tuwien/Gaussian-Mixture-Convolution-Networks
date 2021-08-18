#ifndef CONVOLUTION_FITTING_TREE_H
#define CONVOLUTION_FITTING_TREE_H

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
#include "lbvh/building.h"
#include "lbvh/morton_code.h"
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


    const torch::Tensor m_data;
    const torch::Tensor m_kernels;
    const Config m_config;
    torch::Tensor m_nodes;

    struct Node
    {
        index_type parent_idx; // parent node
        index_type left_idx;   // index of left  child node
        index_type right_idx;  // index of right child node
        index_type object_idx; // == 0xFFFFFFFF if internal node.
    };


    Tree(const torch::Tensor& data, const torch::Tensor& kernels, const Config& config);
    inline torch::Tensor aabb_from_positions(const torch::Tensor& data_positions, const torch::Tensor& kernel_positions) const;
    torch::Tensor compute_morton_codes(const torch::Tensor& data, const torch::Tensor& kernels) const;
    at::Tensor create_tree_nodes(const at::Tensor& morton_codes) const;
};

} // namespace bvh_mhem_fit

#endif // CONVOLUTION_FITTING_TREE_H
