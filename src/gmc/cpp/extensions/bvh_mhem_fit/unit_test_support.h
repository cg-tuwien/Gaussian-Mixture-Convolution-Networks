#ifndef UNIT_TEST_SUPPORT_H
#define UNIT_TEST_SUPPORT_H

#include <cassert>

#include "util/gaussian.h"
#include "util/glm.h"

void assert_similar(float a, float b) {
    auto v = std::abs((a + b) / 2);
    v = gpe::max(v, 1.f);
    assert((std::abs(a - b) / v) < 0.00001f);
}
template<int N_DIMS>
void assert_similar(const glm::vec<N_DIMS, float>& a, const glm::vec<N_DIMS, float>& b) {
    for (int i = 0; i < N_DIMS; ++i) {
        assert_similar(a[i], b[i]);
    }
}
template<int N_DIMS>
void assert_similar(const glm::mat<N_DIMS, N_DIMS, float>& a, const glm::mat<N_DIMS, N_DIMS, float>& b) {
    for (int i = 0; i < N_DIMS; ++i) {
        assert_similar(a[i], b[i]);
    }
}

template<int N_DIMS>
void assert_similar(const gpe::Gaussian<N_DIMS, float>& ad, const gpe::Gaussian<N_DIMS, float>& an) {
    assert_similar(ad.weight, an.weight);
    assert_similar(ad.position, an.position);
    assert_similar(ad.covariance, an.covariance);
}

template<int N_DIMS>
glm::vec<N_DIMS, float> _vec(float a, float b, float c) {
    glm::vec<3, float> r;
    r.x = a;
    r.y = b;
    r.z = c;
    return r;
}

template<int DIMS>
glm::mat<DIMS, DIMS, float> _cov(float xx, float xy, float xz, float yy, float yz, float zz) {
    glm::mat<3, 3, float> r;
    r[0][0] = xx;
    r[0][1] = xy;
    r[0][2] = xz;
    r[1][0] = xy;
    r[1][1] = yy;
    r[1][2] = yz;
    r[2][0] = xz;
    r[2][1] = yz;
    r[2][2] = zz;
    return r;
}

std::vector<float> _scalarCollection() {
    std::vector<float> retval;
    retval.push_back(0);
    retval.push_back(1);
    for (float s = -2.f; s < 2.5f; s += 0.3f) {
        retval.push_back(s);
    }
    return retval;
}

template<int DIMS>
std::vector<glm::vec<DIMS, float>> _vecCollection() {
    std::vector<glm::vec<DIMS, float>> retval;
    retval.push_back(_vec<DIMS>(0, 0, 0));
    retval.push_back(_vec<DIMS>(1, 1, 1));
    retval.push_back(_vec<DIMS>(-1, -1, -1));
    retval.push_back(_vec<DIMS>(1, 2, 3));
    retval.push_back(_vec<DIMS>(-1, -2, -3));
    retval.push_back(_vec<DIMS>(-2.3f, 1.2f, 0.8f));
    retval.push_back(_vec<DIMS>(-3.2f, 0.6f, -0.7f));
    retval.push_back(_vec<DIMS>(0.2f,  -1.4f, 2.5f));
    return retval;
}

template<int DIMS>
std::vector<glm::mat<DIMS, DIMS, float>> _covCollection() {
    std::vector<glm::mat<DIMS, DIMS, float>> retval;
    retval.push_back(glm::mat<DIMS, DIMS, float>(1));
    retval.push_back(glm::mat<DIMS, DIMS, float>(1) * 0.5f);
    retval.push_back(glm::mat<DIMS, DIMS, float>(1) * 2.0f);
    retval.push_back(glm::mat<DIMS, DIMS, float>(1) * 2.0f + 0.5f);
    retval.push_back(glm::mat<DIMS, DIMS, float>(1) * 2.0f - 0.5f);

    retval.push_back(_cov<DIMS>(1.5f, -0.1f,  0.2f, 2.5f, -0.3f, 2.2f));
    retval.push_back(_cov<DIMS>(2.5f,  0.5f, -0.2f, 1.5f,  0.3f, 3.2f));
    return retval;
}

template<int DIMS>
std::vector<gpe::Gaussian<DIMS, float>> _gaussianCollection() {
    using G = gpe::Gaussian<DIMS, float>;
    std::vector<G> collection;

    std::vector<float> weights = {0.f, 0.4f, 1.f};

    std::vector<glm::vec<DIMS, float>> positions;
    positions.push_back(_vec<DIMS>(0, 0, 0));
    positions.push_back(_vec<DIMS>(-1.4f, 2.5, 0.9f));

    std::vector<glm::mat<DIMS, DIMS, float>> covs;
    covs.push_back(glm::mat<DIMS, DIMS, float>(1));
    covs.push_back(glm::mat<DIMS, DIMS, float>(1) * 2.0f - 0.5f);
    covs.push_back(_cov<DIMS>(2.5f,  0.5f, -0.2f, 1.5f,  0.3f, 3.2f));

    for (auto weight : weights) {
        for (auto pos : positions) {
            for (auto cov : covs) {
                collection.push_back(G{weight, pos, cov});
            }
        }
    }
    return collection;
}

template<typename T1, typename T2, typename T3, typename Function, typename GradFunction>
void test_binarycase(T1 a, T2 b, T3 grad, Function fun, GradFunction gradfun) {
    auto a_ad = gpe::makeAutodiff(a);
    auto b_ad = gpe::makeAutodiff(b);
    auto r_ad = fun(a_ad, b_ad);
    gpe::propagateGrad(r_ad, grad);

    T1 grad_a;
    T2 grad_b;
    gradfun(a, b, &grad_a, &grad_b, grad);

    assert_similar(gpe::extractGrad(a_ad), grad_a);
    assert_similar(gpe::extractGrad(b_ad), grad_b);
}

template<typename T1, typename T2, typename Function, typename GradFunction>
void test_unarycase(T1 a, T2 incoming_grad, Function fun, GradFunction gradfun) {
    auto a_ad = gpe::makeAutodiff(a);
    auto result_ad = fun(a_ad);
    gpe::propagateGrad(result_ad, incoming_grad);
    auto grad_analytical = gradfun(a, incoming_grad);
    auto grad_ad = gpe::extractGrad(a_ad);
    assert_similar(grad_ad, grad_analytical);
}

#endif // UNIT_TEST_SUPPORT_H
