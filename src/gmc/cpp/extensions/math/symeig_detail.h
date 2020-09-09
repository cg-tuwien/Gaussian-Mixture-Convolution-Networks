#ifndef MATH_SYMEIG_DETAIL_H
#define MATH_SYMEIG_DETAIL_H

#include <thrust/tuple.h>

#include <glm/glm.hpp>

#include "math/scalar.h"
#include "math/matrix.h"

namespace gpe {

namespace detail {

template <typename scalar_t>
__forceinline__ __host__ __device__ thrust::tuple<glm::vec<2, scalar_t>, glm::mat<2, 2, scalar_t>> compute_symeig(const glm::mat<2, 2, scalar_t>& matrix) {
    using vec = glm::vec<2, scalar_t>;
    using mat = glm::mat<2, 2, scalar_t>;
    // we follow pytorch notation, eigenvectors are in the rows
    // [1] https://math.stackexchange.com/questions/395698/fast-way-to-calculate-eigen-of-2x2-matrix-using-a-formula
    // [2] http://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html
    // [3] https://en.wikipedia.org/wiki/Eigenvalue_algorithm#2.C3.972_matrices

    // using [2], but we want to normalise the eigen vectors
    // our matrices are symmetric, so b == c, and the eigen vectors are orthogonal to each other.
    // normalisation becomes instable for small vectors, so we should take the larger one and rotate it for greater stability

    // for positive semidefinit L1 and L2 (the eigenvalues) are > 0
    // L1 is always larger, because sqrt is positive and L1 = T/2 + sqrt
    // of the 2 formulas in [2], the one subtracting the smaller of a and d is more stable (gives the larger number for normalisation)

    auto T = gpe::trace(matrix);
    auto D = glm::determinant(matrix);
//    std::cout << "D=" << D << std::endl;
    const auto a = matrix[0][0];
    const auto b = matrix[1][0];
    const auto d = matrix[1][1];
//    std::cout << "a=" << a << " b=" << b << " d=" << d << std::endl;
    auto f1 = T / scalar_t(2.0);
    auto f2 = gpe::sqrt(T*T / scalar_t(4.0) - D);
//    std::cout << "f1=" << f1 << " f2=" << f2 << std::endl;
    auto eigenvalues = vec(f1 - f2, f1 + f2);
//    std::cout << "eigenvalues=" << eigenvalues[0] << "/" << eigenvalues[1] << std::endl;
    if (b == scalar_t(0)) {
        return thrust::make_tuple(eigenvalues, mat(1, 0, 0, 1));
    }
    vec e1;
    if (a < d)
        e1 = vec(b, eigenvalues[1]-a);
    else
        e1 = vec(eigenvalues[1]-d, b);
    e1 = glm::normalize(e1);
//    std::cout << "e1=" << e1[0] << "/" << e1[1] << std::endl;
    return thrust::make_tuple(eigenvalues, mat(vec(e1[1], -e1[0]), e1));
}


}

}

#endif // SYMEIG_DETAIL_H
