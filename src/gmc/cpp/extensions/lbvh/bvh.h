#ifndef LBVH_BVH_H
#define LBVH_BVH_H

#include <type_traits>

#include "common.h"
#include "mixture.h"
#include "lbvh/aabb.h"

namespace lbvh
{
namespace detail
{
struct Node
{
    using index_type = uint16_t;
    using index_type_torch = int16_t;   // std::make_unsigned_t<index_type> doesn't work with cuda
    index_type parent_idx; // parent node
    index_type left_idx;   // index of left  child node
    index_type right_idx;  // index of right child node
    index_type object_idx; // == 0xFFFFFFFF if internal node.
};

// a set of pointers to use it on device.
template<typename scalar_t, typename Object, bool IsConst>
struct basic_device_bvh;
template<typename scalar_t, typename Object>
struct basic_device_bvh<scalar_t, Object, false>
{
    using real_type  = scalar_t;
    using aabb_type  = Aabb<real_type>;
    using node_type  = detail::Node;
    using index_type = std::uint32_t;
    using object_type = Object;

    unsigned int num_nodes;   // (# of internal node) + (# of leaves), 2N+1
    unsigned int num_objects; // (# of leaves), the same as the number of objects

    node_type *  nodes;
    aabb_type *  aabbs;
    object_type* objects;
};
template<typename scalar_t, typename Object>
struct basic_device_bvh<scalar_t, Object, true>
{
    using real_type  = scalar_t;
    using aabb_type  = Aabb<real_type>;
    using node_type  = detail::Node;
    using index_type = std::uint32_t;
    using object_type = Object;

    unsigned int num_nodes;  // (# of internal node) + (# of leaves), 2N+1
    unsigned int num_objects;// (# of leaves), the same as the number of objects

    node_type   const* nodes;
    aabb_type   const* aabbs;
    object_type const* objects;
};

template<typename T>
struct TorchTypeMapper;

template<>
struct TorchTypeMapper<int32_t> {
    static inline constexpr torch::ScalarType id() { return torch::ScalarType::Int; }
};

template<>
struct TorchTypeMapper<int64_t> {
    static inline constexpr torch::ScalarType id() { return torch::ScalarType::Long; }
};

template<>
struct TorchTypeMapper<float> {
    static inline constexpr torch::ScalarType id() { return torch::ScalarType::Float; }
};

template<>
struct TorchTypeMapper<double> {
    static inline constexpr torch::ScalarType id() { return torch::ScalarType::Double; }
};
//
} // detail

template<typename scalar_t, typename Object>
using  bvh_device = detail::basic_device_bvh<scalar_t, Object, false>;
template<typename scalar_t, typename Object>
using cbvh_device = detail::basic_device_bvh<scalar_t, Object, true>;

template<typename scalar_t, typename Object>
class Bvh
{
public:
    // rebuild bvh.cpp after changing these types!
    using real_type   = scalar_t;
    using index_type = std::uint32_t;
    using object_type = Object;
    using aabb_type   = Aabb<real_type>;
    using node_type   = detail::Node;

    using morton_torch_t = int64_t;
    using morton_cuda_t = std::make_unsigned<morton_torch_t>::type;


  public:

    Bvh(const torch::Tensor& inversed_mixture)
        : m_mixture(inversed_mixture), m_n(gpe::get_ns(inversed_mixture))
    {
        m_n_leaf_nodes = m_n.components;
        m_n_internal_nodes = m_n_leaf_nodes - 1;
        m_n_nodes = m_n_leaf_nodes * 2 - 1;
        this->construct();
    }

    Bvh(const torch::Tensor& inversed_mixture, const torch::Tensor& nodes, const torch::Tensor& aabbs)
        : m_mixture(inversed_mixture), m_nodes(nodes), m_aabbs(aabbs), m_n(gpe::get_ns(inversed_mixture))
    {
        m_n_leaf_nodes = m_n.components;
        m_n_internal_nodes = m_n_leaf_nodes - 1;
        m_n_nodes = m_n_leaf_nodes * 2 - 1;
    }

    void construct();

protected:
    torch::Tensor compute_aabbs();

    torch::Tensor compute_morton_codes(const torch::Tensor& aabbs, const torch::Tensor& aabb_whole) const;

    std::tuple<torch::Tensor, torch::Tensor> sort_morton_codes(const torch::Tensor& morton_codes, const torch::Tensor& object_aabbs) const;

    torch::Tensor create_leaf_nodes(const torch::Tensor& morton_codes);

    void create_internal_nodes(const torch::Tensor& morton_codes);

    void create_aabbs_for_internal_nodes();

public:
    unsigned m_n_internal_nodes;
    unsigned m_n_leaf_nodes;
    unsigned m_n_nodes;
    const torch::Tensor m_mixture;
    torch::Tensor m_nodes;
    torch::Tensor m_aabbs;
    gpe::MixtureNs m_n;
};





} // lbvh
#endif// LBVH_BVH_H
