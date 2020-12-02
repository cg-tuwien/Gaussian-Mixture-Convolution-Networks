#ifndef MATH_MATRIX_H
#define MATH_MATRIX_H

#include "math/gpe_glm.h"
#include "math/scalar.h"
#include "util/cuda.h"

using uint = unsigned int;

namespace gpe {

template <typename scalar_t>
EXECUTION_DEVICES scalar_t trace(const glm::mat<2, 2, scalar_t>& m) {
    return m[0][0] + m[1][1];
}

template <typename scalar_t>
EXECUTION_DEVICES scalar_t trace(const glm::mat<3, 3, scalar_t>& m) {
    return m[0][0] + m[1][1] + m[2][2];
}

template <int N_DIMS, typename scalar_t>
EXECUTION_DEVICES scalar_t squared_norm(const glm::vec<N_DIMS, scalar_t>& v) {
    return glm::dot(v, v);
}

template <int DIMS, typename scalar_t>
EXECUTION_DEVICES glm::vec<DIMS, scalar_t>&
vec(scalar_t& memory_location) {
    return reinterpret_cast<glm::vec<DIMS, scalar_t>&>(memory_location);
}

template <int DIMS, typename scalar_t>
EXECUTION_DEVICES const glm::vec<DIMS, scalar_t>&
vec(const scalar_t& memory_location) {
    return reinterpret_cast<const glm::vec<DIMS, scalar_t>&>(memory_location);
}

template <int DIMS, typename scalar_t>
EXECUTION_DEVICES const glm::mat<DIMS, DIMS, scalar_t>& mat(const scalar_t& memory_location) {
    return reinterpret_cast<const glm::mat<DIMS, DIMS, scalar_t>&>(memory_location);
}

template <int DIMS, typename scalar_t>
EXECUTION_DEVICES glm::mat<DIMS, DIMS, scalar_t>& mat(scalar_t& memory_location) {
    return reinterpret_cast<glm::mat<DIMS, DIMS, scalar_t>&>(memory_location);
}

template <int DIMS, typename scalar_t>
EXECUTION_DEVICES bool isnan(const glm::vec<DIMS, scalar_t>& x) {
    bool nan = false;
    for (unsigned i = 0; i < DIMS; ++i)
        nan = nan || gpe::isnan(x[i]);
    return nan;
}

template <int DIMS, typename scalar_t>
EXECUTION_DEVICES bool isnan(const glm::mat<DIMS, DIMS, scalar_t>& x) {
    bool nan = false;
    for (unsigned i = 0; i < DIMS; ++i)
        for (unsigned j = 0; j < DIMS; ++j)
            nan = nan || gpe::isnan(x[i][j]);
    return nan;
}

template <int DIMS, typename scalar_t>
EXECUTION_DEVICES glm::vec<DIMS, scalar_t> diagonal(const glm::mat<DIMS, DIMS, scalar_t>& x) {
    glm::vec<DIMS, scalar_t> d;
    for (unsigned i = 0; i < DIMS; ++i)
        d[i] = x[i][i];
    return d;
}

}

#endif // HELPERS_H
