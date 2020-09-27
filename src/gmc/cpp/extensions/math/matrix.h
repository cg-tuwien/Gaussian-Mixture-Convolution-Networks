#ifndef MATH_MATRIX_H
#define MATH_MATRIX_H

#include <glm/glm.hpp>

#ifndef __CUDACC__
#define __device__
#define __host__
#endif

#ifndef __forceinline__
#define __forceinline__ inline
#endif

using uint = unsigned int;

namespace gpe {

template <typename scalar_t>
__forceinline__ __host__ __device__ scalar_t trace(const glm::mat<2, 2, scalar_t>& m) {
    return m[0][0] + m[1][1];
}

template <typename scalar_t>
__forceinline__ __host__ __device__ scalar_t trace(const glm::mat<3, 3, scalar_t>& m) {
    return m[0][0] + m[1][1] + m[2][2];
}

template <int N_DIMS, typename scalar_t>
__forceinline__ __host__ __device__ scalar_t squared_norm(const glm::vec<N_DIMS, scalar_t>& v) {
    return glm::dot(v, v);
}

template <int DIMS, typename scalar_t>
__forceinline__ __host__ __device__ glm::vec<DIMS, scalar_t>&
vec(scalar_t& memory_location) {
    return reinterpret_cast<glm::vec<DIMS, scalar_t>&>(memory_location);
}

template <int DIMS, typename scalar_t>
__forceinline__ __host__ __device__ const glm::vec<DIMS, scalar_t>&
vec(const scalar_t& memory_location) {
    return reinterpret_cast<const glm::vec<DIMS, scalar_t>&>(memory_location);
}

template <int DIMS, typename scalar_t>
__forceinline__ __host__ __device__ const glm::mat<DIMS, DIMS, scalar_t>& mat(const scalar_t& memory_location) {
    return reinterpret_cast<const glm::mat<DIMS, DIMS, scalar_t>&>(memory_location);
}

template <int DIMS, typename scalar_t>
__forceinline__ __host__ __device__ glm::mat<DIMS, DIMS, scalar_t>& mat(scalar_t& memory_location) {
    return reinterpret_cast<glm::mat<DIMS, DIMS, scalar_t>&>(memory_location);
}

}

#endif // HELPERS_H
