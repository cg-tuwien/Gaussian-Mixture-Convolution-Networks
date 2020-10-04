#ifndef LBVH_QUERY_H
#define LBVH_QUERY_H

#include "predicator.h"
#include "bvh.h"

#define LBVH_N_QUERY_THREADS 32
#define LBVH_QUERY_STACK_SIZE 17

namespace lbvh
{
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

} // lbvh
#endif// LBVH_QUERY_H
