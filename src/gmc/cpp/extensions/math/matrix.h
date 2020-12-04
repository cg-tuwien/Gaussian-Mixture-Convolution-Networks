#ifndef MATH_MATRIX_H
#define MATH_MATRIX_H

#include "math/gpe_glm.h"
#include "math/scalar.h"
#include "util/cuda.h"

using uint = unsigned int;

namespace gpe {

namespace {
// transpose of adjugate matrix, needed for grad_determinant, which is the cofactor matrix
// https://en.wikipedia.org/wiki/Jacobi%27s_formula
// https://en.wikipedia.org/wiki/Adjugate_matrix
template <typename scalar_t>
EXECUTION_DEVICES glm::mat<2, 2, scalar_t> cofactor(const glm::mat<2, 2, scalar_t>& m) {
    return {{m[1][1], -m[1][0]}, {-m[0][1], m[0][0]}};
}

// reduces nicely: https://godbolt.org/z/M5MhET
template <typename scalar_t>
EXECUTION_DEVICES  scalar_t detexcl(const glm::mat<3, 3, scalar_t>& m, unsigned excl_i, unsigned excl_j) {
    // map 0 -> 1, 2 in bits 00 -> 01, 10
    //     1 -> 0, 2         01 -> 00, 10
    //     2 -> 0, 1         10 -> 00, 01
    const auto i1 = unsigned(excl_i < 1);
    const auto i2 = 2 - (excl_i >> 1);

    // same again
    const auto j1 = unsigned(excl_j < 1);
    const auto j2 = 2 - (excl_j >> 1);
    return m[i1][j1] * m[i2][j2] - m[i1][j2] * m[i2][j1];
}
template <typename scalar_t>
EXECUTION_DEVICES glm::mat<3, 3, scalar_t> cofactor(const glm::mat<3, 3, scalar_t>& m) {
    glm::mat<3, 3, scalar_t> cof;
    for (unsigned i = 0; i < 3; ++i) {
        for (unsigned j = 0; j < 3; ++j) {
            const auto sign = ((i ^ j) % 2 == 0) ? 1 : -1;
            cof[i][j] = sign * detexcl(m, i, j);
        }
    }
    return cof;
}
}

template <typename scalar_t, int DIMS>
EXECUTION_DEVICES glm::mat<DIMS, DIMS, scalar_t> grad_determinant(const glm::mat<DIMS, DIMS, scalar_t>& cov, scalar_t grad) {
    assert(glm::determinant(cov) > 0);
    return cofactor(cov) * grad;
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

}

#endif // HELPERS_H
