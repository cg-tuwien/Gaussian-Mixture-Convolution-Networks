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

template <int DIMS, typename scalar_t>
__forceinline__ __host__ __device__ typename std::conditional<std::is_const<scalar_t>::value, const glm::vec<DIMS, std::remove_cv_t<scalar_t>>, glm::vec<DIMS, scalar_t>>::type&
vec(scalar_t& memory_location) {
    return reinterpret_cast<typename std::conditional<std::is_const<scalar_t>::value, const glm::vec<DIMS, std::remove_cv_t<scalar_t>>, glm::vec<DIMS, scalar_t>>::type&>(memory_location);
}

template <int DIMS, typename scalar_t>
__forceinline__ __host__ __device__ typename std::conditional<std::is_const<scalar_t>::value, const glm::mat<DIMS, DIMS, std::remove_cv_t<scalar_t>>, glm::mat<DIMS, DIMS, scalar_t>>::type&
mat(scalar_t& memory_location) {
    return reinterpret_cast<typename std::conditional<std::is_const<scalar_t>::value, const glm::mat<DIMS, DIMS, std::remove_cv_t<scalar_t>>, glm::mat<DIMS, DIMS, scalar_t>>::type&>(memory_location);
}

}

#endif // HELPERS_H
