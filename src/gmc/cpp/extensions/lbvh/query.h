#ifndef LBVH_QUERY_H
#define LBVH_QUERY_H

#include "predicator.h"
#include "bvh.h"

#define LBVH_N_QUERY_THREADS 32
#define LBVH_QUERY_STACK_SIZE 17

namespace lbvh
{
// query object indices that potentially overlaps with query aabb.
//
// requirements:
// - OutputIterator should be writable and its object_type should be uint32_t
//
template<typename scalar_t, typename Objects, bool IsConst, typename OutputIterator, typename Predicate>
__device__
unsigned int query_device(
        const detail::basic_device_bvh<scalar_t, Objects, IsConst>& bvh,
        const Predicate& predicate, OutputIterator outiter,
        const unsigned int max_buffer_size = 0xFFFFFFFF) noexcept
{
    using bvh_type   = detail::basic_device_bvh<scalar_t, Objects, IsConst>;
    using index_type = typename bvh_type::index_type;
    using aabb_type  = typename bvh_type::aabb_type;
    using node_type  = typename bvh_type::node_type;

    index_type  stack[64]; // is it okay?
    index_type* stack_ptr = stack;
    *stack_ptr++ = 0; // root node is always 0

    unsigned int num_found = 0;
    do
    {
        const index_type node  = *--stack_ptr;
        const index_type L_idx = bvh.nodes[node].left_idx;
        const index_type R_idx = bvh.nodes[node].right_idx;

        if(predicate(bvh.aabbs[L_idx]))
        {
            const auto obj_idx = bvh.nodes[L_idx].object_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(num_found < max_buffer_size)
                {
                    *outiter++ = obj_idx;
                }
                ++num_found;
            }
            else // the node is not a leaf.
            {
                *stack_ptr++ = L_idx;
            }
        }
        if(predicate(bvh.aabbs[R_idx]))
        {
            const auto obj_idx = bvh.nodes[R_idx].object_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(num_found < max_buffer_size)
                {
                    *outiter++ = obj_idx;
                }
                ++num_found;
            }
            else // the node is not a leaf.
            {
                *stack_ptr++ = R_idx;
            }
        }
    }
    while (stack < stack_ptr);
    return num_found;
}

template<typename scalar_t, typename Objects, bool IsConst, typename Predicate, typename Function>
__device__
void query_device_with_fun(const detail::basic_device_bvh<scalar_t, Objects, IsConst>& bvh,
                           const Predicate& predicate,
                           Function fun) noexcept
{
    using bvh_type   = detail::basic_device_bvh<scalar_t, Objects, IsConst>;
    using index_type = uint16_t;
    using aabb_type  = typename bvh_type::aabb_type;
    using node_type  = typename bvh_type::node_type;

    __shared__ index_type stack[LBVH_N_QUERY_THREADS][LBVH_QUERY_STACK_SIZE]; // is it okay?
    index_type* stack_ptr = stack[threadIdx.x];
    *stack_ptr++ = 0; // root node is always 0

    do
    {
        const index_type node  = *--stack_ptr;
        const index_type L_idx = bvh.nodes[node].left_idx;
        const index_type R_idx = bvh.nodes[node].right_idx;

        if(predicate(bvh.aabbs[L_idx]))
        {
            const auto obj_idx = bvh.nodes[L_idx].object_idx;
            if(obj_idx != node_type::index_type(0xFFFFFFFF))
            {
                fun(obj_idx);
            }
            else // the node is not a leaf.
            {
                *stack_ptr++ = L_idx;
            }
        }
        if(predicate(bvh.aabbs[R_idx]))
        {
            const auto obj_idx = bvh.nodes[R_idx].object_idx;
            if(obj_idx != node_type::index_type(0xFFFFFFFF))
            {
                fun(obj_idx);
            }
            else // the node is not a leaf.
            {
                *stack_ptr++ = R_idx;
            }
        }
    }
    while (stack[threadIdx.x] < stack_ptr);
}

// query object index that is the nearst to the query point.
//
// requirements:
// - DistanceCalculator must be able to calc distance between a point to an object.
//
//template<typename scalar_t, typename Objects, bool IsConst,
//         typename DistanceCalculator>
//__device__
//thrust::pair<unsigned int, scalar_t> query_device(
//        const detail::basic_device_bvh<scalar_t, Objects, IsConst>& bvh,
//        const query_nearest<scalar_t>& q, DistanceCalculator calc_dist) noexcept
//{
//    using bvh_type   = detail::basic_device_bvh<scalar_t, Objects, IsConst>;
//    using real_type  = typename bvh_type::real_type;
//    using index_type = typename bvh_type::index_type;
//    using aabb_type  = typename bvh_type::aabb_type;
//    using node_type  = typename bvh_type::node_type;

//    // pair of {node_idx, mindist}
//    thrust::pair<index_type, real_type>  stack[64];
//    thrust::pair<index_type, real_type>* stack_ptr = stack;
//    *stack_ptr++ = thrust::make_pair(0, mindist(bvh.aabbs[0], q.target));

//    unsigned int nearest = 0xFFFFFFFF;
//    real_type dist_to_nearest_object = infinity<real_type>();
//    do
//    {
//        const auto node  = *--stack_ptr;
//        if(node.second > dist_to_nearest_object)
//        {
//            // if aabb mindist > already_found_mindist, it cannot have a nearest
//            continue;
//        }

//        const index_type L_idx = bvh.nodes[node.first].left_idx;
//        const index_type R_idx = bvh.nodes[node.first].right_idx;

//        const aabb_type& L_box = bvh.aabbs[L_idx];
//        const aabb_type& R_box = bvh.aabbs[R_idx];

//        const real_type L_mindist = mindist(L_box, q.target);
//        const real_type R_mindist = mindist(R_box, q.target);

//        const real_type L_minmaxdist = minmaxdist(L_box, q.target);
//        const real_type R_minmaxdist = minmaxdist(R_box, q.target);

//        // there should be an object that locates within minmaxdist.

//        if(L_mindist <= R_minmaxdist) // L is worth considering
//        {
//            const auto obj_idx = bvh.nodes[L_idx].object_idx;
//            if(obj_idx != 0xFFFFFFFF) // leaf node
//            {
//                const real_type dist = calc_dist(q.target, bvh.objects[obj_idx]);
//                if(dist <= dist_to_nearest_object)
//                {
//                    dist_to_nearest_object = dist;
//                    nearest = obj_idx;
//                }
//            }
//            else
//            {
//                *stack_ptr++ = thrust::make_pair(L_idx, L_mindist);
//            }
//        }
//        if(R_mindist <= L_minmaxdist) // R is worth considering
//        {
//            const auto obj_idx = bvh.nodes[R_idx].object_idx;
//            if(obj_idx != 0xFFFFFFFF) // leaf node
//            {
//                const real_type dist = calc_dist(q.target, bvh.objects[obj_idx]);
//                if(dist <= dist_to_nearest_object)
//                {
//                    dist_to_nearest_object = dist;
//                    nearest = obj_idx;
//                }
//            }
//            else
//            {
//                *stack_ptr++ = thrust::make_pair(R_idx, R_mindist);
//            }
//        }
//        assert(stack_ptr < stack + 64);
//    }
//    while (stack < stack_ptr);
//    return thrust::make_pair(nearest, dist_to_nearest_object);
//}

//template<typename scalar_t, typename Objects, typename AABBGetter,
//         typename MortonCodeCalculator, typename OutputIterator>
//unsigned int query_host(
//    const bvh<scalar_t, Objects, AABBGetter, MortonCodeCalculator>& tree,
//    const query_overlap<scalar_t> q, OutputIterator outiter,
//    const unsigned int max_buffer_size = 0xFFFFFFFF)
//{
//    using bvh_type   = ::lbvh::bvh<scalar_t, Objects, AABBGetter, MortonCodeCalculator>;
//    using index_type = typename bvh_type::index_type;
//    using aabb_type  = typename bvh_type::aabb_type;
//    using node_type  = typename bvh_type::node_type;

//    if(!tree.query_host_enabled())
//    {
//        throw std::runtime_error("lbvh::bvh query_host is not enabled");
//    }

//    std::vector<std::size_t> stack;
//    stack.reserve(64);
//    stack.push_back(0);

//    unsigned int num_found = 0;
//    do
//    {
//        const index_type node  = stack.back(); stack.pop_back();
//        const index_type L_idx = tree.nodes_host()[node].left_idx;
//        const index_type R_idx = tree.nodes_host()[node].right_idx;

//        if(intersects(q.target, tree.aabbs_host()[L_idx]))
//        {
//            const auto obj_idx = tree.nodes_host()[L_idx].object_idx;
//            if(obj_idx != 0xFFFFFFFF)
//            {
//                if(num_found < max_buffer_size)
//                {
//                    *outiter++ = obj_idx;
//                }
//                ++num_found;
//            }
//            else // the node is not a leaf.
//            {
//                stack.push_back(L_idx);
//            }
//        }
//        if(intersects(q.target, tree.aabbs_host()[R_idx]))
//        {
//            const auto obj_idx = tree.nodes_host()[R_idx].object_idx;
//            if(obj_idx != 0xFFFFFFFF)
//            {
//                if(num_found < max_buffer_size)
//                {
//                    *outiter++ = obj_idx;
//                }
//                ++num_found;
//            }
//            else // the node is not a leaf.
//            {
//                stack.push_back(R_idx);
//            }
//        }
//    }
//    while (!stack.empty());
//    return num_found;
//}

//template<typename scalar_t, typename Objects, typename AABBGetter,
//         typename MortonCodeCalculator, typename DistanceCalculator>
//std::pair<unsigned int, scalar_t> query_host(
//        const bvh<scalar_t, Objects, AABBGetter, MortonCodeCalculator>& tree,
//        const query_nearest<scalar_t>& q, DistanceCalculator calc_dist) noexcept
//{
//    using bvh_type   = ::lbvh::bvh<scalar_t, Objects, AABBGetter, MortonCodeCalculator>;
//    using real_type  = typename bvh_type::real_type;
//    using index_type = typename bvh_type::index_type;
//    using aabb_type  = typename bvh_type::aabb_type;
//    using node_type  = typename bvh_type::node_type;

//    if(!tree.query_host_enabled())
//    {
//        throw std::runtime_error("lbvh::bvh query_host is not enabled");
//    }

//    // pair of {node_idx, mindist}
//    std::vector<std::pair<index_type, real_type>> stack = {
//        {0, mindist(tree.aabbs_host()[0], q.target)}
//    };
//    stack.reserve(64);

//    unsigned int nearest = 0xFFFFFFFF;
//    real_type current_nearest_dist = infinity<real_type>();
//    do
//    {
//        const auto node = stack.back(); stack.pop_back();
//        if(node.second > current_nearest_dist)
//        {
//            // if aabb mindist > already_found_mindist, it cannot have a nearest
//            continue;
//        }

//        const index_type L_idx = tree.nodes_host()[node.first].left_idx;
//        const index_type R_idx = tree.nodes_host()[node.first].right_idx;

//        const aabb_type& L_box = tree.aabbs_host()[L_idx];
//        const aabb_type& R_box = tree.aabbs_host()[R_idx];

//        const real_type L_mindist = mindist(L_box, q.target);
//        const real_type R_mindist = mindist(R_box, q.target);

//        const real_type L_minmaxdist = minmaxdist(L_box, q.target);
//        const real_type R_minmaxdist = minmaxdist(R_box, q.target);

//       // there should be an object that locates within minmaxdist.

//        if(L_mindist <= R_minmaxdist) // L is worth considering
//        {
//            const auto obj_idx = tree.nodes_host()[L_idx].object_idx;
//            if(obj_idx != 0xFFFFFFFF) // leaf node
//            {
//                const real_type dist = calc_dist(q.target, tree.objects_host()[obj_idx]);
//                if(dist <= current_nearest_dist)
//                {
//                    current_nearest_dist = dist;
//                    nearest = obj_idx;
//                }
//            }
//            else
//            {
//                stack.emplace_back(L_idx, L_mindist);
//            }
//        }
//        if(R_mindist <= L_minmaxdist) // R is worth considering
//        {
//            const auto obj_idx = tree.nodes_host()[R_idx].object_idx;
//            if(obj_idx != 0xFFFFFFFF) // leaf node
//            {
//                const real_type dist = calc_dist(q.target, tree.objects_host()[obj_idx]);
//                if(dist <= current_nearest_dist)
//                {
//                    current_nearest_dist = dist;
//                    nearest = obj_idx;
//                }
//            }
//            else
//            {
//                stack.emplace_back(R_idx, R_mindist);
//            }
//        }
//    }
//    while (!stack.empty());
//    return std::make_pair(nearest, current_nearest_dist);
//}
} // lbvh
#endif// LBVH_QUERY_H
