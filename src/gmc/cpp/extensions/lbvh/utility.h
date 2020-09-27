#ifndef LBVH_UTILITY_H
#define LBVH_UTILITY_H
#include <numeric>
#include <vector_types.h>
#include <math_constants.h>

#include <glm/glm.hpp>

namespace lbvh
{

template<typename T> struct vector_of;
template<> struct vector_of<float>  {using type = float4;};
template<> struct vector_of<double> {using type = double4;};

template <typename scalar_t> __device__
typename vector_of<scalar_t>::type make_vector_of(const glm::vec<2, scalar_t>& glm_vec) {
    return {glm_vec.x, glm_vec.y, scalar_t(0), scalar_t(0)};
}

template <typename scalar_t> __device__
typename vector_of<scalar_t>::type make_vector_of(const glm::vec<3, scalar_t>& glm_vec) {
    return {glm_vec.x, glm_vec.y, glm_vec.z, scalar_t(0)};
}

template<typename T>
using vector_of_t = typename vector_of<T>::type;

#ifdef __CUDACC__
template<typename T>
__device__
inline T infinity() noexcept;
template<>
__device__
inline float  infinity<float >() noexcept {return CUDART_INF_F;}
template<>
__device__
inline double infinity<double>() noexcept {return CUDART_INF;}
#else
template<typename T>
inline T infinity() noexcept {
    return std::numeric_limits<T>::infinity();
}
#endif

} // lbvh
#endif// LBVH_UTILITY_H
