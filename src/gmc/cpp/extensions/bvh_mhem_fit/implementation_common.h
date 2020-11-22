#ifndef BVH_MHEM_FIT_IMPLEMENTATION_COMMON
#define BVH_MHEM_FIT_IMPLEMENTATION_COMMON

#include <torch/types.h>

#include "BvhMhemFitConfig.h"
#include "containers.h"
#include "lbvh/bvh.h"
#include "hacked_accessor.h"

#define EXECUTION_DEVICES __host__ __device__ __forceinline__

namespace bvh_mhem_fit {

using node_index_torch_t = lbvh::detail::Node::index_type_torch;
using node_index_t = lbvh::detail::Node::index_type;
using gaussian_index_t = uint16_t;
using gaussian_index_torch_t = int16_t;
using Node  = lbvh::detail::Node;

template <typename scalar_t = float>
struct Epsilon {
    static constexpr scalar_t small = scalar_t(0.00000000000000000000000000000000001);
    static constexpr scalar_t large = scalar_t(0.0000000000000001);
    static EXECUTION_DEVICES scalar_t clip(scalar_t v) { return gpe::max(v, small); }
};

template<>
struct Epsilon<double> {
    static constexpr double small = 0.0000000000000000000000000000000000000000000000000000000000000000000001;
    static constexpr double large = 0.00000000000000000000000000001;
    static EXECUTION_DEVICES double clip(double v) { return gpe::max(v, small); }
};


template <typename scalar_t>
struct UIntOfSize {
    using type = uint32_t;
};

template <>
struct UIntOfSize<double> {
    using type = uint64_t;
};

template<typename scalar_t, int N_DIMS, int REDUCTION_N>
struct AugmentedBvh
{
    using aabb_type  = lbvh::Aabb<scalar_t>;
    using Gaussian_type = gpe::Gaussian<N_DIMS, scalar_t>;
    struct NodeAttributes {
        using UIntType = typename UIntOfSize<scalar_t>::type;
        gpe::Vector<Gaussian_type, REDUCTION_N, UIntType> gaussians;
        scalar_t gm_integral;
        UIntType n_child_leaves;
        // when adding an attribute, remember to update the line
        // auto node_attributes = torch::zeros({n_mixtures, n_nodes, REDUCTION_N * mixture.size(-1) + 3}, torch::TensorOptions(mixture.device()).dtype(mixture.scalar_type()));
    };
//    static_assert (alignof (Gaussian_type) == 4, "adsf");
    static_assert (sizeof (NodeAttributes) <= sizeof(scalar_t) * (REDUCTION_N * (1 + N_DIMS + N_DIMS * N_DIMS) + 3), "NodeAttribute is too large and won't fit into the torch::Tensor");
    static_assert (sizeof (NodeAttributes) == sizeof(scalar_t) * (REDUCTION_N * (1 + N_DIMS + N_DIMS * N_DIMS) + 3), "NodeAttribute has unexpected size (it could be smaller, no problem, just unexpected)");

    const unsigned n_internal_nodes;
    const unsigned n_leaves;
    const unsigned n_nodes;
    const int32_t _padding = 0;

    const Node* nodes;                         // size = n_nodes
    const aabb_type* aabbs;                         // size = n_nodes
    Gaussian_type* gaussians;                       // size = n_leaves
    NodeAttributes* per_node_attributes;            // size = n_nodes

    EXECUTION_DEVICES
    AugmentedBvh(int mixture_id,
                 const gpe::PackedTensorAccessor32<node_index_torch_t, 3> nodes,
                 const gpe::PackedTensorAccessor32<scalar_t, 3> aabbs,
                 gpe::PackedTensorAccessor32<scalar_t, 3> mixture,
                 gpe::PackedTensorAccessor32<scalar_t, 3> node_attributes,
                 const gpe::MixtureNs n, const unsigned n_internal_nodes, const unsigned n_nodes)
        : n_internal_nodes(n_internal_nodes), n_leaves(unsigned(n.components)), n_nodes(n_nodes),
          nodes(reinterpret_cast<const Node*>(&nodes[mixture_id][0][0])),
          aabbs(reinterpret_cast<const aabb_type*>(&aabbs[mixture_id][0][0])),
          gaussians(reinterpret_cast<Gaussian_type*>(&mixture[mixture_id][0][0])),
          per_node_attributes(reinterpret_cast<NodeAttributes*>(&node_attributes[mixture_id][0][0]))
    {}

    EXECUTION_DEVICES gpe::Vector<Gaussian_type, REDUCTION_N * 2> collect_child_gaussians(const lbvh::detail::Node* node, scalar_t weight_threshold) const {
        assert(node->left_idx != node_index_t(-1));
        assert(node->right_idx != node_index_t(-1));
        gpe::Vector<Gaussian_type, REDUCTION_N * 2> retval;
        auto condition = [weight_threshold](const Gaussian_type& g) { return gpe::abs(g.weight) >= weight_threshold; };
        retval.push_back_if(per_node_attributes[node->left_idx].gaussians, condition);
        retval.push_back_if(per_node_attributes[node->right_idx].gaussians, condition);
        return retval;
    }

    EXECUTION_DEVICES node_index_t node_id(const lbvh::detail::Node* node) {
        auto id = node_index_t(node - nodes);
        assert(id < n_nodes);
        return id;
    }
};

} // namespace bvh_mhem_fit

#endif
