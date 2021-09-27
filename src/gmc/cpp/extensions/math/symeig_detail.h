#ifndef MATH_SYMEIG_DETAIL_H
#define MATH_SYMEIG_DETAIL_H

#include <thrust/tuple.h>

#include "util/epsilon.h"
#include "util/glm.h"
#include "util/scalar.h"

namespace gpe {

namespace detail {

// These algorithms might have some numerical stability issues lurking, but it currently it doesn't matter for gmcn since they are used
// only to compute the BVH in case of evaluation.

template <typename scalar_t> __forceinline__ __host__ __device__
thrust::tuple<glm::vec<2, scalar_t>, glm::mat<2, 2, scalar_t>> compute_symeig(const glm::mat<2, 2, scalar_t>& matrix) {
    using vec = glm::vec<2, scalar_t>;
    using mat = glm::mat<2, 2, scalar_t>;
    // we follow pytorch notation, eigenvectors are column vectors
    // [1] https://math.stackexchange.com/questions/395698/fast-way-to-calculate-eigen-of-2x2-matrix-using-a-formula
    // [2] http://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html
    // [3] https://en.wikipedia.org/wiki/Eigenvalue_algorithm#2.C3.972_matrices

    // using [2], but we want to normalise the eigen vectors
    // our matrices are symmetric, so b == c, and the eigen vectors are orthogonal to each other.
    // normalisation becomes instable for small vectors, so we should take the larger one and rotate it for greater stability

    // for positive semidefinit L1 and L2 (the eigenvalues) are > 0
    // L1 is always larger, because sqrt is positive and L1 = T/2 + sqrt
    // of the 2 formulas in [2], the one subtracting the smaller of a and d is more stable (gives the larger number for normalisation)

    const scalar_t T = gpe::trace(matrix);
    const scalar_t D = glm::determinant(matrix);
//    std::cout << "D=" << D << std::endl;
    const scalar_t a = matrix[0][0];
    const scalar_t b = matrix[1][0];
    const scalar_t d = matrix[1][1];
//    std::cout << "a=" << a << " b=" << b << " d=" << d << std::endl;
    const scalar_t f1 = T / scalar_t(2.0);
    const scalar_t f2 = gpe::sqrt(gpe::abs(T*T / scalar_t(4.0) - D));
//    std::cout << "f1=" << f1 << " f2=" << f2 << std::endl;
    const auto eigenvalues = vec(f1 - f2, f1 + f2);
//    std::cout << "eigenvalues=" << eigenvalues[0] << "/" << eigenvalues[1] << std::endl;
    if (b == scalar_t(0)) {
        return thrust::make_tuple(eigenvalues, mat(1, 0, 0, 1));
    }

    vec e0, e1;
    if (a < d) {
        e0 = vec(b, eigenvalues[0]-a);
        e1 = vec(b, eigenvalues[1]-a);
    }
    else {
        e0 = vec(eigenvalues[0]-d, b);
        e1 = vec(eigenvalues[1]-d, b);
    }
    e0 = glm::normalize(e0);
    e1 = glm::normalize(e1);
//    std::cout << "e1=" << e1[0] << "/" << e1[1] << std::endl;
    return thrust::make_tuple(eigenvalues, glm::transpose(mat(e0, e1)));

//    vec e1;
//    if (a < d)
//        e1 = vec(b, eigenvalues[1]-a);
//    else
//        e1 = vec(eigenvalues[1]-d, b);
//    e1 = glm::normalize(e1);
////    std::cout << "e1=" << e1[0] << "/" << e1[1] << std::endl;
//    return thrust::make_tuple(eigenvalues, mat(vec(e1[1], -e1[0]), e1));
}

template <typename scalar_t> __forceinline__ __host__ __device__
glm::vec<3, scalar_t> normAndScaleLongestVector(const glm::mat<3, 3, scalar_t>& A) {
    scalar_t max_length = 0;
    unsigned max_index = unsigned(-1);
    for (unsigned i = 0; i < 3; ++i) {
        const auto length = gpe::squared_norm(A[i]);
        if (length > max_length) {
            max_length = length;
            max_index = i;
        }
    }
    assert(max_index < 3);
    max_length = gpe::sqrt(max_length);
    return A[max_index] / max_length;
}

// not tested for non-symmetric matrices: look at normAndScaleLongestVector, it might need a transpose.
template <typename scalar_t> __forceinline__ __host__ __device__
thrust::tuple<glm::vec<3, scalar_t>, glm::mat<3, 3, scalar_t>> compute_symeig(const glm::mat<3, 3, scalar_t>& A) {
    using Vec = glm::vec<3, scalar_t>;
    using Mat = glm::mat<3, 3, scalar_t>;
    // we follow pytorch notation, eigenvectors are in the rows
    // [1] https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3%C3%973_matrices
    // [2] https://dl.acm.org/doi/10.1145/355578.366316

    scalar_t eig0, eig1, eig2;
    const auto p1 = gpe::sq(A[0][1]) + gpe::sq(A[0][2]) + gpe::sq(A[1][2]);
    if (p1 <= gpe::Epsilon<scalar_t>::large) {
        // A is diagonal.
        eig0 = A[0][0];
        eig1 = A[1][1];
        eig2 = A[2][2];
    }
    else {
        const auto q = gpe::trace(A)/3;
        const auto p2 = gpe::sq(A[0][0] - q) + gpe::sq(A[1][1] - q) + gpe::sq(A[2][2] - q) + 2 * p1;
        const auto p = gpe::sqrt(p2 / 6);
        const auto B = (1 / p) * (A - q * Mat(1));
        const auto r = glm::determinant(B) / 2;

        // In exact arithmetic for a symmetric matrix  -1 <= r <= 1
        // but computation error can leave it slightly outside this range.
        scalar_t phi;
        if (r <= -1)
            phi = glm::pi<scalar_t>() / 3;
        else if (r >= 1)
            phi = 0;
        else
            phi = gpe::acos(r) / 3;

        // the eigenvalues satisfy eig2 <= eig1 <= eig0
        eig0 = q + 2 * p * gpe::cos(phi);
        eig2 = q + 2 * p * gpe::cos(phi + (2 * glm::pi<scalar_t>() / 3));
        eig1 = 3 * q - eig0 - eig2;         // since trace(A) = eig1 + eig2 + eig3;
    }

    const auto eigvec0 = normAndScaleLongestVector((A - eig1 * Mat(1)) * (A - eig2 * Mat(1)));
    const auto eigvec1 = normAndScaleLongestVector((A - eig0 * Mat(1)) * (A - eig2 * Mat(1)));
//    const auto eigvec2 = normAndScaleLongestVector((A - eig0 * Mat(1)) * (A - eig1 * Mat(1)));
//    assert(glm::dot(eigvec0, eigvec1) < scalar_t(0.0000001));
    const auto eigvec2 = glm::cross(eigvec0, eigvec1);  // symmetric matrices have orthogonal eigenvectors.

    // torch expects reverse order
    return {{eig2, eig1, eig0}, glm::transpose(Mat{eigvec2, eigvec1, eigvec0})};
}


}

}

#endif // SYMEIG_DETAIL_H
