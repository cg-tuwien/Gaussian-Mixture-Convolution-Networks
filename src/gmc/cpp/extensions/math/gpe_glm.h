#ifndef MATH_GPE_GLM_H
#define MATH_GPE_GLM_H

//#define GLM_FORCE_SIZE_T_LENGTH
#define GLM_FORCE_PURE
#define GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_INLINE
#define GLM_FORCE_UNRESTRICTED_GENTYPE
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>


namespace gpe {

template <typename scalar_t>
GLM_FUNC_QUALIFIER scalar_t trace(const glm::mat<2, 2, scalar_t>& m) {
    return m[0][0] + m[1][1];
}

template <typename scalar_t>
GLM_FUNC_QUALIFIER scalar_t trace(const glm::mat<3, 3, scalar_t>& m) {
    return m[0][0] + m[1][1] + m[2][2];
}

template <typename scalar_t>
GLM_FUNC_QUALIFIER scalar_t sum(const glm::vec<2, scalar_t>& v) {
    return v[0] + v[1];
}

template <typename scalar_t>
GLM_FUNC_QUALIFIER scalar_t sum(const glm::vec<3, scalar_t>& v) {
    return v[0] + v[1] + v[2];
}

template <typename scalar_t>
GLM_FUNC_QUALIFIER scalar_t sum(const glm::mat<2, 2, scalar_t>& m) {
    return sum(m[0]) + sum(m[1]);
}

template <typename scalar_t>
GLM_FUNC_QUALIFIER scalar_t sum(const glm::mat<3, 3, scalar_t>& m) {
    return sum(m[0]) + sum(m[1]) + sum(m[2]);
}

template <typename scalar_t>
GLM_FUNC_QUALIFIER glm::mat<2, 2, scalar_t> cwise_mul(const glm::mat<2, 2, scalar_t>& m1, const glm::mat<2, 2, scalar_t>& m2) {
    return {m1[0] * m2[0], m1[1] * m2[1]};
}

template <typename scalar_t>
GLM_FUNC_QUALIFIER glm::mat<3, 3, scalar_t> cwise_mul(const glm::mat<3, 3, scalar_t>& m1, const glm::mat<3, 3, scalar_t>& m2) {
    return {m1[0] * m2[0], m1[1] * m2[1], m1[2] * m2[2]};
}

template <int N_DIMS, typename scalar_t>
GLM_FUNC_QUALIFIER scalar_t squared_norm(const glm::vec<N_DIMS, scalar_t>& v) {
    return glm::dot(v, v);
}

template <int DIMS, typename scalar_t>
GLM_FUNC_QUALIFIER glm::vec<DIMS, scalar_t> diagonal(const glm::mat<DIMS, DIMS, scalar_t>& x) {
    glm::vec<DIMS, scalar_t> d;
    for (unsigned i = 0; i < DIMS; ++i)
        d[i] = x[i][i];
    return d;
}

} // namespace gpe

#endif // MATH_GPE_GLM_H
