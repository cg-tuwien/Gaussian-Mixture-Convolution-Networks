#ifndef LBVH_PREDICATOR_H
#define LBVH_PREDICATOR_H
#include "utility.h"
#include "aabb.h"

namespace lbvh
{

template<typename scalar_t>
struct query_inside
{
    __device__ __host__
    query_inside(const typename vector_of<scalar_t>::type& point): point(point) {}

    query_inside()  = default;
    ~query_inside() = default;
    query_inside(const query_inside&) = default;
    query_inside(query_inside&&)      = default;
    query_inside& operator=(const query_inside&) = default;
    query_inside& operator=(query_inside&&)      = default;

    __device__ __host__
    inline bool operator()(const Aabb<scalar_t>& box) const noexcept
    {
        return inside(point, box);
    }

    typename vector_of<scalar_t>::type point;
};

//template<typename scalar_t>
//__device__ __host__
//query_inside<scalar_t> inside_aabb(const typename vector_of<scalar_t>::type& point) noexcept
//{
//    return query_inside<scalar_t>(point);
//}

__device__ __host__
query_inside<float> inside_aabb(const float4& point) noexcept
{
    return query_inside<float>(point);
}

__device__ __host__
query_inside<double> inside_aabb(const double4& point) noexcept
{
    return query_inside<double>(point);
}

template<typename scalar_t>
struct query_overlap
{
    __device__ __host__
    query_overlap(const Aabb<scalar_t>& tgt): target(tgt) {}

    query_overlap()  = default;
    ~query_overlap() = default;
    query_overlap(const query_overlap&) = default;
    query_overlap(query_overlap&&)      = default;
    query_overlap& operator=(const query_overlap&) = default;
    query_overlap& operator=(query_overlap&&)      = default;

    __device__ __host__
    inline bool operator()(const Aabb<scalar_t>& box) noexcept
    {
        return intersects(box, target);
    }

    Aabb<scalar_t> target;
};

template<typename scalar_t>
__device__ __host__
query_overlap<scalar_t> overlaps(const Aabb<scalar_t>& region) noexcept
{
    return query_overlap<scalar_t>(region);
}

template<typename scalar_t>
struct query_nearest
{
    // float4/double4
    using vector_type = typename vector_of<scalar_t>::type;

    __device__ __host__
    query_nearest(const vector_type& tgt): target(tgt) {}

    query_nearest()  = default;
    ~query_nearest() = default;
    query_nearest(const query_nearest&) = default;
    query_nearest(query_nearest&&)      = default;
    query_nearest& operator=(const query_nearest&) = default;
    query_nearest& operator=(query_nearest&&)      = default;

    vector_type target;
};

__device__ __host__
inline query_nearest<float> nearest(const float4& point) noexcept
{
    return query_nearest<float>(point);
}
__device__ __host__
inline query_nearest<float> nearest(const float3& point) noexcept
{
    return query_nearest<float>(make_float4(point.x, point.y, point.z, 0.0f));
}
__device__ __host__
inline query_nearest<double> nearest(const double4& point) noexcept
{
    return query_nearest<double>(point);
}
__device__ __host__
inline query_nearest<double> nearest(const double3& point) noexcept
{
    return query_nearest<double>(make_double4(point.x, point.y, point.z, 0.0));
}

} // lbvh
#endif// LBVH_PREDICATOR_H
