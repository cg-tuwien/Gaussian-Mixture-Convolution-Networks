#ifndef EM_FITTING_COMMON_H
#define EM_FITTING_COMMON_H

#include <glm/gtc/constants.hpp>
#include "common.h"

template <typename scalar_t, int DIMS>
__forceinline__ __host__ __device__ scalar_t gaussian_amplitude(const glm::mat<DIMS, DIMS, scalar_t>& inversed_cov) {
    constexpr auto a = scalar_t(std::pow(2. * glm::pi<double>(), - DIMS * 0.5));
    if (glm::determinant(inversed_cov) == 0) {
        std::cout << "err: singular cov" << std::endl;
        return 1;
    }
    return a * gm::sqrt(glm::determinant(inversed_cov));
}

template <typename scalar_t>
__forceinline__ __host__ __device__ scalar_t trace(const glm::mat<2, 2, scalar_t>& m) {
    return m[0][0] + m[1][1];
}

template <typename scalar_t>
__forceinline__ __host__ __device__ scalar_t trace(const glm::mat<3, 3, scalar_t>& m) {
    return m[0][0] + m[1][1] + m[2][2];
}

#endif // EM_FITTING_COMMON_H
