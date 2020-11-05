#ifndef LBVH_MORTON_CODE_H
#define LBVH_MORTON_CODE_H
#include <vector_types.h>
#include <cuda_runtime.h>
#include <cstdint>

#include <glm/glm.hpp>

#include "math/scalar.h"
#include "cuda_operations.h"

namespace lbvh
{

__device__ __host__
inline std::uint32_t expand_bits3(std::uint32_t v) noexcept
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

template <typename T>
inline T copy_or(T v, unsigned shift) noexcept {
    return (v << shift) | v;
}

std::uint64_t expand_bits2(std::uint32_t v) noexcept
{
    uint64_t t = v;
    t = copy_or(t, 16) & 0x0000'FFFF'0000'FFFFul;
    t = copy_or(t, 8)  & 0x00FF'00FF'00FF'00FFul;
    t = copy_or(t, 4)  & 0x0F0F'0F0F'0F0F'0F0Ful;
    t = copy_or(t, 2)  & 0x3333'3333'3333'3333ul;
    t = copy_or(t, 1)  & 0x5555'5555'5555'5555ul;
    return t;
}

std::uint32_t expand_bits2(std::uint16_t v) noexcept
{
    uint32_t t = v;
    t = copy_or(t, 8)  & 0x00FF'00FFu;
    t = copy_or(t, 4)  & 0x0F0F'0F0Fu;
    t = copy_or(t, 2)  & 0x3333'3333u;
    t = copy_or(t, 1)  & 0x5555'5555u;
    return t;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__device__ __host__
inline std::uint32_t morton_code(float4 xyz, float resolution = 1024.0f) noexcept
{
    xyz.x = gpe::min(gpe::max(xyz.x * resolution, 0.0f), resolution - 1.0f);
    xyz.y = gpe::min(gpe::max(xyz.y * resolution, 0.0f), resolution - 1.0f);
    xyz.z = gpe::min(gpe::max(xyz.z * resolution, 0.0f), resolution - 1.0f);
    const std::uint32_t xx = expand_bits3(static_cast<std::uint32_t>(xyz.x));
    const std::uint32_t yy = expand_bits3(static_cast<std::uint32_t>(xyz.y));
    const std::uint32_t zz = expand_bits3(static_cast<std::uint32_t>(xyz.z));
    return xx * 4 + yy * 2 + zz;
}

__device__ __host__
inline std::uint32_t morton_code(double4 xyz, double resolution = 1024.0) noexcept
{
    xyz.x = gpe::min(gpe::max(xyz.x * resolution, 0.0), resolution - 1.0);
    xyz.y = gpe::min(gpe::max(xyz.y * resolution, 0.0), resolution - 1.0);
    xyz.z = gpe::min(gpe::max(xyz.z * resolution, 0.0), resolution - 1.0);
    const std::uint32_t xx = expand_bits3(static_cast<std::uint32_t>(xyz.x));
    const std::uint32_t yy = expand_bits3(static_cast<std::uint32_t>(xyz.y));
    const std::uint32_t zz = expand_bits3(static_cast<std::uint32_t>(xyz.z));
    return xx * 4 + yy * 2 + zz;
}

template<typename scalar_t>
__device__ __host__
// 2 left most bits are 0
inline std::uint32_t morton_code(const glm::vec<3, scalar_t>& vec, uint32_t resolution = 1024) {
    uint32_t x = gpe::min(gpe::max(uint32_t(vec.x * resolution), 0u), resolution - 1);
    uint32_t y = gpe::min(gpe::max(uint32_t(vec.y * resolution), 0u), resolution - 1);
    uint32_t z = gpe::min(gpe::max(uint32_t(vec.z * resolution), 0u), resolution - 1);
    auto xx = expand_bits3(x);
    auto yy = expand_bits3(y);
    auto zz = expand_bits3(z);
    return (xx << 2) + (yy << 1) + zz;
}


template<typename scalar_t>
__device__ __host__
inline std::uint64_t morton_code(uint16_t component_id, const glm::vec<3, scalar_t>& pos, const glm::vec<3, scalar_t>& cov_diag, scalar_t resolution = 1024.0) {
    /// old
//    uint32_t morton_pos = morton_code(pos);         // 30 bits
//    uint64_t result = morton_pos;
//    result = (result << 16) | component_id;

    /// cov 1
    // layout:
    // we use 30 bits of position morton code, 18 bits for cov_diag morton code and 16 bits for component_id
    // in the result:
    // 12 bits only position, 18 + 18 bits interleaved position with cov_diag, and 16 bits component id

//    uint32_t morton_pos = morton_code(pos);         // 30 bits
//    uint32_t morton_cov = morton_code(cov_diag);    // 30 bits
//    uint64_t result = 0;
//    result |= morton_pos >> 18;
//    morton_pos = morton_pos & 0xFFFu;               // 12 bits
//    morton_cov = morton_cov >> 18;                  // 12 most significant bits
//    morton_pos = expand_bits2(uint16_t(morton_pos));
//    morton_cov = expand_bits2(uint16_t(morton_cov));
//    uint32_t morton_poscov = (morton_pos << 1) | morton_cov;    // 24 bits
//    assert((morton_poscov & 0xFF'FFFF) == morton_poscov);
//    result = (result << 24) | morton_poscov;
//    result = (result << 16) | component_id;

    /// cov 2
    // layout:
    // here we use 27 bits of position morton code, 27 bits for cov_diag morton code and 10 bits for component_id
    // in the result:
    // 27 + 27 bits interleaved position with cov_diag, and 10 bits component id
    uint32_t morton_pos = morton_code(pos);         // 30 bits
    uint32_t morton_cov = morton_code(cov_diag);    // 30 bits
    morton_pos = expand_bits2(uint16_t(morton_pos));
    morton_cov = expand_bits2(uint16_t(morton_cov));
    auto morton_poscov = (morton_pos << 1) | morton_cov;    // 60 bits
    uint64_t result = morton_poscov << 4;
    result = result & (~0x3FFu);
    assert (componend_id < 1024);
    result = result | component_id;

    /// cov 3, second
    // layout:
    // here we use 27 bits of position morton code, 27 bits for cov_diag morton code and 10 bits for component_id
    // in the result:
    // 28 bits position, then 28 bits cov_diag, and 10 bits component id
//    uint32_t morton_pos = morton_code(pos);         // 30 bits
//    uint32_t morton_cov = morton_code(cov_diag);    // 30 bits
//    uint64_t result = (morton_pos & (~0x1F)) << 34;
//    assert((result & 0xFFFF'FFE0'0000'0000u) == result);
//    assert((result >> (64 - 27)) == (morton_pos >> 3));

//    result |= (morton_cov & (~0x1F)) << 9;
//    assert((result & 0xFFFF'FFFF'FFFF'FC00u) == result);
//    assert((((result << 27) >> 27) >> (64 - 27)) == (morton_cov >> 3));
//    assert (componend_id < 1024);
//    result = result | (component_id & 0x3FF);


    /// cov 4, first
    // layout:
    // here we use 27 bits of position morton code, 27 bits for cov_diag morton code and 10 bits for component_id
    // in the result:
    // 28 bits position, then 28 bits cov_diag, and 10 bits component id
//    uint32_t morton_pos = morton_code(pos);         // 30 bits
//    uint32_t morton_cov = morton_code(cov_diag);    // 30 bits
//    uint64_t result = (morton_cov & (~0x1F)) << 34;
//    assert((result & 0xFFFF'FFE0'0000'0000u) == result);
//    assert((result >> (64 - 27)) == (morton_cov >> 3));

//    result |= (morton_pos & (~0x1F)) << 9;
//    assert((result & 0xFFFF'FFFF'FFFF'FC00u) == result);
//    assert((((result << 27) >> 27) >> (64 - 27)) == (morton_pos >> 3));
//    assert (componend_id < 1024);
//    result = result | (component_id & 0x3FF);

    return result;
}

__host__ __device__ __forceinline__
int common_upper_bits(const uint32_t lhs, const uint32_t rhs) noexcept
{
    return gpe::clz(lhs ^ rhs);
}
__host__ __device__ __forceinline__
int common_upper_bits(const uint64_t lhs, const uint64_t rhs) noexcept
{
    return gpe::clz(lhs ^ rhs);
}

} // lbvh
#endif// LBVH_MORTON_CODE_H
