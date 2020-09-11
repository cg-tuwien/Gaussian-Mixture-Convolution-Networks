#ifndef LBVH_PREDICATOR_H
#define LBVH_PREDICATOR_H
#include "utility.h"
#include "aabb.h"

namespace lbvh
{

template<typename Real>
struct query_inside
{
    __device__ __host__
    query_inside(const typename vector_of<Real>::type& point): point(point) {}

    query_inside()  = default;
    ~query_inside() = default;
    query_inside(const query_inside&) = default;
    query_inside(query_inside&&)      = default;
    query_inside& operator=(const query_inside&) = default;
    query_inside& operator=(query_inside&&)      = default;

    __device__ __host__
    inline bool operator()(const aabb<Real>& box) const noexcept
    {
        return inside(point, box);
    }

    typename vector_of<Real>::type point;
};

//template<typename Real>
//__device__ __host__
//query_inside<Real> inside_aabb(const typename vector_of<Real>::type& point) noexcept
//{
//    return query_inside<Real>(point);
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

template<typename Real>
struct query_overlap
{
    __device__ __host__
    query_overlap(const aabb<Real>& tgt): target(tgt) {}

    query_overlap()  = default;
    ~query_overlap() = default;
    query_overlap(const query_overlap&) = default;
    query_overlap(query_overlap&&)      = default;
    query_overlap& operator=(const query_overlap&) = default;
    query_overlap& operator=(query_overlap&&)      = default;

    __device__ __host__
    inline bool operator()(const aabb<Real>& box) noexcept
    {
        return intersects(box, target);
    }

    aabb<Real> target;
};

template<typename Real>
__device__ __host__
query_overlap<Real> overlaps(const aabb<Real>& region) noexcept
{
    return query_overlap<Real>(region);
}

template<typename Real>
struct query_nearest
{
    // float4/double4
    using vector_type = typename vector_of<Real>::type;

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
