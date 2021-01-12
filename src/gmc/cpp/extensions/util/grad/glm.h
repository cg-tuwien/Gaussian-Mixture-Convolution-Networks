#ifndef GPE_UTIL_GRAD_GLM_H
#define GPE_UTIL_GRAD_GLM_H

#include "util/glm.h"
#include "util/cuda.h"

namespace gpe {

namespace grad {

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
EXECUTION_DEVICES glm::mat<DIMS, DIMS, scalar_t> determinant(const glm::mat<DIMS, DIMS, scalar_t>& cov, scalar_t grad) {
    assert(glm::determinant(cov) > 0);
    return cofactor(cov) * grad;
}

template <typename scalar_t, int DIMS>
EXECUTION_DEVICES glm::mat<DIMS, DIMS, scalar_t> inverse(const glm::mat<DIMS, DIMS, scalar_t>& cov, const glm::mat<DIMS, DIMS, scalar_t>& incoming_grad) {
    assert(glm::determinant(cov) > 0);
    // acording to https://people.maths.ox.ac.uk/gilesm/files/AD2008.pdf. omitting the transpose as we have symmetric matrices
    const auto cov_inv = glm::inverse(cov);
    return - cov_inv * incoming_grad * cov_inv;
}

template <typename scalar_t, int DIMS>
EXECUTION_DEVICES glm::mat<DIMS, DIMS, scalar_t> inverse_with_cached_covInv(const glm::mat<DIMS, DIMS, scalar_t>& cached_covInv, const glm::mat<DIMS, DIMS, scalar_t>& incoming_grad) {
    // acording to https://people.maths.ox.ac.uk/gilesm/files/AD2008.pdf. omitting the transpose as we have symmetric matrices
    return - cached_covInv * incoming_grad * cached_covInv;
}

template<int N_DIMS, typename T>
EXECUTION_DEVICES
void outerProduct(const glm::vec<N_DIMS, T>& a, const glm::vec<N_DIMS, T>& b, glm::vec<N_DIMS, T>* a_grad, glm::vec<N_DIMS, T>* b_grad, const glm::mat<N_DIMS, N_DIMS, T>& incoming_grad) {
    for (int i = 0; i < N_DIMS; ++i) {
        (*a_grad)[i] = 0;
        (*b_grad)[i] = 0;
    }
    for (int i = 0; i < N_DIMS; ++i) {
        for (int j = 0; j < N_DIMS; ++j) {
            (*a_grad)[i] += incoming_grad[j][i] * b[j];
            (*b_grad)[j] += incoming_grad[j][i] * a[i];
        }
    }
}

template<int N_DIMS, typename T>
EXECUTION_DEVICES
void dot(const glm::vec<N_DIMS, T>& a, const glm::vec<N_DIMS, T>& b, glm::vec<N_DIMS, T>* a_grad, glm::vec<N_DIMS, T>* b_grad, const T& incoming_grad) {
    *a_grad = b * incoming_grad;
    *b_grad = a * incoming_grad;
}


} // namespace grad

} // namespace gpe

#endif // GPE_UTIL_GRAD_MATRIX_H
