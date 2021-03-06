#ifndef UNIT_TEST_SUPPORT_H
#define UNIT_TEST_SUPPORT_H

#include <cassert>
#include <vector>

#include <catch2/catch.hpp>
#include <torch/types.h>

#include "util/autodiff.h"
#include "util/gaussian.h"
#include "util/glm.h"

template<typename scalar_t>
bool are_similar(scalar_t a, scalar_t b, scalar_t precision = scalar_t(0.00001)) {
    auto v = std::abs((a + b) / 2);
    v = gpe::max(v, scalar_t(1));
    return((std::abs(a - b) / v) < precision);
}
template<int N_DIMS, typename scalar_t>
bool are_similar(const glm::vec<N_DIMS, scalar_t>& a, const glm::vec<N_DIMS, scalar_t>& b, scalar_t precision = scalar_t(0.00001)) {
    bool similar = true;
    for (int i = 0; i < N_DIMS; ++i) {
        similar = similar && are_similar(a[i], b[i], precision);
    }
    return similar;
}
template<int N_DIMS, typename scalar_t>
bool are_similar(const glm::mat<N_DIMS, N_DIMS, scalar_t>& a, const glm::mat<N_DIMS, N_DIMS, scalar_t>& b, scalar_t precision = scalar_t(0.00001)) {
    bool similar = true;
    for (int i = 0; i < N_DIMS; ++i) {
        similar = similar && are_similar(a[i], b[i], precision);
    }
    return similar;
}

template<int N_DIMS, typename scalar_t>
bool are_similar(const gpe::Gaussian<N_DIMS, scalar_t>& ad, const gpe::Gaussian<N_DIMS, scalar_t>& an, scalar_t precision = scalar_t(0.00001)) {
    return are_similar(ad.weight, an.weight, precision)
            && are_similar(ad.position, an.position, precision)
            && are_similar(ad.covariance, an.covariance, precision);
}

template<int N_DIMS, typename scalar_t = float>
glm::vec<N_DIMS, scalar_t> _vec(scalar_t a, scalar_t b, scalar_t c) {
    glm::vec<3, scalar_t> r;
    r.x = a;
    r.y = b;
    r.z = c;
    return r;
}

template<int DIMS, typename scalar_t = float>
glm::mat<DIMS, DIMS, scalar_t> _cov(scalar_t xx, scalar_t xy, scalar_t xz, scalar_t yy, scalar_t yz, scalar_t zz) {
    glm::mat<3, 3, scalar_t> r;
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

template <typename scalar_t = float>
std::vector<scalar_t> _scalarCollection() {
    std::vector<scalar_t> retval;
    retval.push_back(0);
    retval.push_back(1);
    for (scalar_t s = scalar_t(-2.); s < scalar_t(2.5); s += scalar_t(0.3)) {
        retval.push_back(s);
    }
    return retval;
}

template <typename scalar_t = float>
std::vector<scalar_t> _smallPositiveScalarCollection() {
    std::vector<scalar_t> retval;
    retval.push_back(0);
    retval.push_back(1);
    retval.push_back(0.7f);
    retval.push_back(1.55f);
    return retval;
}

template <typename scalar_t = float>
std::vector<scalar_t> _smallNegativeScalarCollection() {
    std::vector<scalar_t> retval;
    retval.push_back(0);
    retval.push_back(-1);
    retval.push_back(-0.7f);
    retval.push_back(-1.55f);
    return retval;
}

template<int DIMS, typename scalar_t = float>
std::vector<glm::vec<DIMS, scalar_t>> _vecCollection() {
    std::vector<glm::vec<DIMS, scalar_t>> retval;
    retval.push_back(_vec<DIMS, scalar_t>(0, 0, 0));
    retval.push_back(_vec<DIMS, scalar_t>(1, 1, 1));
    retval.push_back(_vec<DIMS, scalar_t>(-1, -1, -1));
    retval.push_back(_vec<DIMS, scalar_t>(1, 2, 3));
    retval.push_back(_vec<DIMS, scalar_t>(-1, -2, -3));
    retval.push_back(_vec<DIMS, scalar_t>(-2.3f, 1.2f, 0.8f));
    retval.push_back(_vec<DIMS, scalar_t>(-3.2f, 0.6f, -0.7f));
    retval.push_back(_vec<DIMS, scalar_t>(0.2f,  -1.4f, 2.5f));
    return retval;
}

template<int DIMS, typename scalar_t = float>
std::vector<glm::vec<DIMS, scalar_t>> _smallVecCollection() {
    std::vector<glm::vec<DIMS, scalar_t>> retval;
    retval.push_back(_vec<DIMS, scalar_t>(0, 0, 0));
    retval.push_back(_vec<DIMS, scalar_t>(1, 1, 1));
    retval.push_back(_vec<DIMS, scalar_t>(-1, -1, -1));
    retval.push_back(_vec<DIMS, scalar_t>(-2.3f, 1.2f, 0.8f));
    return retval;
}

template<int DIMS, typename scalar_t = float>
std::vector<glm::mat<DIMS, DIMS, scalar_t>> _covCollection() {
    std::vector<glm::mat<DIMS, DIMS, scalar_t>> retval;
    retval.push_back(glm::mat<DIMS, DIMS, scalar_t>(1));
    retval.push_back(glm::mat<DIMS, DIMS, scalar_t>(1) * scalar_t(0.5));
    retval.push_back(glm::mat<DIMS, DIMS, scalar_t>(1) * scalar_t(2.0));
    retval.push_back(glm::mat<DIMS, DIMS, scalar_t>(1) * scalar_t(2.0) + scalar_t(0.5));
    retval.push_back(glm::mat<DIMS, DIMS, scalar_t>(1) * scalar_t(2.0) - scalar_t(0.5));

    retval.push_back(_cov<DIMS, scalar_t>(1.5f, -0.1f,  0.2f, 2.5f, -0.3f, 2.2f));
    retval.push_back(_cov<DIMS, scalar_t>(2.5f,  0.5f, -0.2f, 1.5f,  0.3f, 3.2f));
    return retval;
}

template<int DIMS, typename scalar_t = float>
std::vector<glm::mat<DIMS, DIMS, scalar_t>> _smallCovCollection() {
    std::vector<glm::mat<DIMS, DIMS, scalar_t>> retval;
    retval.push_back(glm::mat<DIMS, DIMS, scalar_t>(1));
    retval.push_back(glm::mat<DIMS, DIMS, scalar_t>(1) * scalar_t(2.0) + scalar_t(0.5));
    retval.push_back(glm::mat<DIMS, DIMS, scalar_t>(1) * scalar_t(2.0) - scalar_t(0.5));
    retval.push_back(_cov<DIMS, scalar_t>(2.5f,  0.5f, -0.2f, 1.5f,  0.3f, 3.2f));
    return retval;
}

template<int DIMS>
std::vector<gpe::Gaussian<DIMS, float>> _gaussianCollection() {
    using G = gpe::Gaussian<DIMS, float>;
    std::vector<G> collection;

    std::vector<float> weights = {0.f, 0.4f, 1.f};

    std::vector<glm::vec<DIMS, float>> positions;
    positions.push_back(_vec<DIMS>(0.f, 0.f, 0.f));
    positions.push_back(_vec<DIMS>(-1.4f, 2.5f, 0.9f));

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


template<int DIMS>
std::vector<gpe::Gaussian<DIMS, float>> _gaussianCollection2() {
    using G = gpe::Gaussian<DIMS, float>;
    std::vector<G> collection;

    std::vector<float> weights = {0.f, -0.4f, 1.6f};

    std::vector<glm::vec<DIMS, float>> positions;
    positions.push_back(_vec<DIMS>(0, 0, 0));
    positions.push_back(_vec<DIMS>(-1.4f, 2.5f, 0.9f));

    std::vector<glm::mat<DIMS, DIMS, float>> covs;
    covs.push_back(glm::mat<DIMS, DIMS, float>(1));
    covs.push_back(glm::mat<DIMS, DIMS, float>(1) * 2.0f - 0.5f);
    covs.push_back(_cov<DIMS>(2.5f,  0.5f, -0.2f, 1.5f,  0.3f, 3.2f));
    covs.push_back(_cov<DIMS>(1.0f,  0.0f, 0.0f, 0.0f, 0.0f, 0.0f));
    covs.push_back(_cov<DIMS>(0.0f,  0.0f, 0.0f, 1.0f, 0.0f, 0.0f));
    covs.push_back(_cov<DIMS>(0.0f,  1.0f, 0.0f, 0.0f, 0.0f, 0.0f));

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

    REQUIRE(are_similar(gpe::extractGrad(a_ad), grad_a));
    REQUIRE(are_similar(gpe::extractGrad(b_ad), grad_b));
}

template<typename T1, typename T2, typename Function, typename GradFunction>
void test_unarycase(T1 a, T2 incoming_grad, Function fun, GradFunction gradfun) {
    auto a_ad = gpe::makeAutodiff(a);
    auto result_ad = fun(a_ad);
    gpe::propagateGrad(result_ad, incoming_grad);
    auto grad_analytical = gradfun(a, incoming_grad);
    auto grad_ad = gpe::extractGrad(a_ad);
    REQUIRE(are_similar(grad_ad, grad_analytical));
}

inline
std::vector<torch::Tensor> _collectionOf2d2GsGrads () {
    std::vector<torch::Tensor> grads;
    grads.emplace_back(torch::tensor({{1.1f,  1.2f,  1.3f,  1.4f,  1.5f,  1.5f,  1.7f},
                                      {1.8f,  1.9f,  0.1f,  0.2f,  0.4f,  0.4f,  5.5f}}).view({1, 1, 2, 7}));

    grads.emplace_back(torch::tensor({{0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f}}).view({1, 1, 2, 7}));

    grads.emplace_back(torch::tensor({{1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f}}).view({1, 1, 2, 7}));

    grads.emplace_back(torch::tensor({{0.0f,  1.0f,  1.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {0.0f,  1.0f,  1.0f,  0.0f,  0.0f,  0.0f,  0.0f}}).view({1, 1, 2, 7}));

    grads.emplace_back(torch::tensor({{0.0f,  0.0f,  0.0f,  1.0f,  1.0f,  1.0f,  1.0f},
                                      {0.0f,  0.0f,  0.0f,  1.0f,  1.0f,  1.0f,  1.0f}}).view({1, 1, 2, 7}));

    grads.emplace_back(torch::tensor({{1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f},
                                      {0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f}}).view({1, 1, 2, 7}));

    grads.emplace_back(torch::tensor({{0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f}}).view({1, 1, 2, 7}));

    grads.emplace_back(torch::tensor({{1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f}}).view({1, 1, 2, 7}));

    grads.emplace_back(torch::tensor({{-1.0f,  -1.0f,  -1.0f,  -1.0f,  -1.0f,  -1.0f,  -1.0f},
                                      {-1.0f,  -1.0f,  -1.0f,  -1.0f,  -1.0f,  -1.0f,  -1.0f}}).view({1, 1, 2, 7}));

    grads.emplace_back(torch::tensor({{-1.1f,  -1.2f,  -1.3f,   1.4f,  -1.5f,  -1.5f,   1.7f},
                                      {-1.8f,   1.9f,   0.1f,  -0.2f,   0.4f,   0.4f,  -5.5f}}).view({1, 1, 2, 7}));
    return grads;
}

inline
std::vector<torch::Tensor> _collectionOf2d4GsGrads () {
    std::vector<torch::Tensor> grads;
    grads.emplace_back(torch::tensor({{1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f}}).view({1, 1, 4, 7}));

    grads.emplace_back(torch::tensor({{-1.0f,  -1.0f,  -1.0f,  -1.0f,  -1.0f,  -1.0f,  -1.0f},
                                      {-1.0f,  -1.0f,  -1.0f,  -1.0f,  -1.0f,  -1.0f,  -1.0f},
                                      {-1.0f,  -1.0f,  -1.0f,  -1.0f,  -1.0f,  -1.0f,  -1.0f},
                                      {-1.0f,  -1.0f,  -1.0f,  -1.0f,  -1.0f,  -1.0f,  -1.0f}}).view({1, 1, 4, 7}));

    grads.emplace_back(torch::tensor({{ 1.1f,   1.2f,   1.3f,   1.4f,   1.5f,   1.5f,   1.7f},
                                      { 1.8f,   1.9f,   0.1f,   0.2f,   0.4f,   0.4f,   5.5f},
                                      { 0.7f,   0.9f,   1.2f,   1.3f,   0.5f,   0.5f,   0.7f},
                                      { 1.0f,   1.1f,   1.6f,   1.4f,   1.4f,   1.4f,   1.2f}}).view({1, 1, 4, 7}));

    grads.emplace_back(torch::tensor({{-1.1f,   1.2f,  -1.3f,   1.4f,  -1.5f,  -1.5f,   1.7f},
                                      { 1.8f,  -1.9f,   0.1f,  -0.2f,   0.4f,   0.4f,  -5.5f},
                                      {-0.7f,   0.9f,  -1.2f,   1.3f,  -0.5f,  -0.5f,   0.7f},
                                      { 1.0f,  -1.1f,   1.6f,  -1.4f,   1.4f,   1.4f,  -1.2f}}).view({1, 1, 4, 7}));
    return grads;
}

inline
std::vector<torch::Tensor> _collectionOf2d8GsGrads () {
    std::vector<torch::Tensor> grads;
    grads.emplace_back(torch::tensor({{1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f}}).view({1, 1, 8, 7}));

    grads.emplace_back(torch::tensor({{-1.1f,   1.2f,  -1.3f,   1.4f,  -0.5f,  -0.5f,   1.7f},
                                      { 1.8f,  -1.9f,   0.1f,  -0.2f,   0.4f,   0.4f,  -5.5f},
                                      {-0.7f,   0.9f,  -1.2f,   1.3f,  -0.5f,  -0.5f,   0.7f},
                                      { 1.0f,  -1.1f,   1.6f,  -1.4f,   0.4f,   0.4f,  -1.2f},
                                      {-1.1f,   1.2f,  -1.3f,   1.4f,  -0.1f,  -0.1f,   1.7f},
                                      { 1.8f,  -1.9f,   0.1f,  -0.2f,   0.4f,   0.4f,  -5.5f},
                                      {-0.7f,   0.9f,  -1.2f,   1.3f,  -0.5f,  -0.5f,   0.7f},
                                      { 1.0f,  -1.1f,   1.6f,  -1.4f,   0.2f,   0.2f,  -1.2f}}).view({1, 1, 8, 7}));

    grads.emplace_back(torch::zeros({1, 1, 8, 7}));
    return grads;
}


inline
std::vector<torch::Tensor> _collectionOf2d32GsGradsWellBehaving () {
    std::vector<torch::Tensor> grads;
    grads.emplace_back(torch::tensor({{1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f}}).view({1, 1, 32, 7}));

    grads.emplace_back(torch::tensor({{ 0.6f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {-1.2f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      { 0.1f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {-0.2f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      { 0.7f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      { 1.4f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {-2.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      { 1.8f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {-0.8f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {-0.6f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      { 1.4f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      { 1.3f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {-0.5f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {-0.6f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {-2.1f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {-0.8f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      { 1.3f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      { 0.4f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      { 1.7f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      { 0.2f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      { 0.5f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {-1.8f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      { 1.3f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {-0.9f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {-0.5f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {-2.3f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      { 0.2f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      { 0.7f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {-0.8f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      {-1.5f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      { 2.2f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                      { 1.6f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f}}).view({1, 1, 32, 7}));
    return grads;
}

inline
std::vector<torch::Tensor> _collectionOf2d32GsGradsExploding () {
    std::vector<torch::Tensor> grads;
    grads.emplace_back(torch::tensor({{1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f},
                                      {1.0f,  1.0f,  1.0f,  1.0f,  0.1f,  0.1f,  1.0f}}).view({1, 1, 32, 7}));

    grads.emplace_back(torch::tensor({{-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f},
                                      {-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f},
                                      {-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f},
                                      {-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f},
                                      {-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f},
                                      {-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f},
                                      {-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f},
                                      {-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f},
                                      {-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f},
                                      {-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f},
                                      {-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f},
                                      {-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f},
                                      {-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f},
                                      {-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f},
                                      {-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f},
                                      {-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f},
                                      {-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f},
                                      {-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f},
                                      {-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f},
                                      {-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f},
                                      {-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f},
                                      {-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f},
                                      {-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f},
                                      {-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f},
                                      {-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f},
                                      {-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f},
                                      {-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f},
                                      {-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f},
                                      {-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f},
                                      {-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f},
                                      {-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f},
                                      {-1.3f,  -1.3f,  -1.3f,  -1.3f,  -.13f,  -.13f,  -1.3f}}).view({1, 1, 32, 7}));

    grads.emplace_back(torch::tensor({{ 1.1f,   1.2f,   1.3f,   1.4f,   1.5f,   1.5f,   1.7f},
                                      { 1.8f,   1.9f,   0.1f,   2.2f,   0.3f,   0.4f,   5.5f},
                                      { 0.7f,   0.4f,   0.2f,   5.3f,   0.5f,   0.5f,   0.7f},
                                      { 1.0f,   3.1f,   1.6f,   0.4f,   1.4f,   1.4f,   1.2f},
                                      {-1.1f,   1.2f,  -1.3f,   1.4f,  -1.5f,  -1.5f,   2.7f},
                                      { 1.8f,  -1.9f,   0.1f,  -0.2f,   0.4f,   0.4f,  -3.5f},
                                      {-0.9f,   0.9f,  -1.2f,   5.3f,  -0.5f,  -0.5f,   0.7f},
                                      { 1.0f,  -1.1f,   1.6f,  -1.4f,   1.4f,   1.4f,  -1.2f},
                                      { 1.1f,   1.2f,   1.3f,   1.4f,   1.5f,   1.5f,   1.7f},
                                      { 1.8f,   1.9f,   0.1f,  -0.2f,  -0.4f,  -0.4f,   5.5f},
                                      { 0.7f,   0.9f,   1.2f,   1.3f,   1.5f,   0.5f,   2.7f},
                                      { 1.0f,   1.1f,   1.6f,   2.6f,   1.4f,   1.4f,   1.2f},
                                      {-1.1f,   2.2f,  -1.3f,   1.4f,  -1.5f,  -1.5f,   1.7f},
                                      { 1.8f,  -1.9f,   0.1f,  -0.9f,   0.4f,   0.4f,  -5.5f},
                                      {-0.7f,   0.9f,  -1.2f,   1.3f,  -0.5f,  -0.5f,   0.7f},
                                      { 1.5f,  -1.1f,   1.6f,  -1.4f,   1.4f,   1.4f,  -1.2f},
                                      { 1.1f,   1.2f,   1.1f,   1.4f,   1.5f,   1.5f,   1.7f},
                                      { 2.8f,   1.9f,   2.1f,   0.2f,   0.7f,   0.7f,   5.5f},
                                      { 0.7f,   0.9f,   1.2f,   1.3f,   0.5f,   0.5f,   0.7f},
                                      { 1.0f,   1.1f,   1.6f,   1.4f,   1.4f,   1.4f,   1.2f},
                                      {-1.1f,   1.2f,  -1.3f,   1.4f,  -1.5f,  -1.5f,   1.7f},
                                      { 1.8f,  -1.9f,   0.1f,  -1.2f,   0.4f,   0.4f,  -2.5f},
                                      {-2.7f,   0.9f,  -1.2f,   2.3f,  -0.9f,  -0.9f,   0.7f},
                                      { 1.0f,  -1.5f,   1.6f,  -1.4f,   1.4f,   1.4f,  -1.2f},
                                      { 0.1f,   1.2f,   3.3f,  -2.4f,   1.5f,   1.5f,   1.7f},
                                      { 1.8f,   1.9f,   3.1f,   1.2f,   0.4f,   0.4f,   6.5f},
                                      { 0.7f,   0.9f,   1.2f,   1.3f,   0.5f,   0.7f,   0.7f},
                                      { 1.0f,   1.1f,   0.6f,   2.4f,   1.4f,   1.4f,   1.2f},
                                      {-3.1f,   1.2f,  -1.3f,   1.4f,  -1.3f,  -1.3f,   1.7f},
                                      { 1.8f,  -1.4f,   0.1f,  -0.2f,   0.4f,   0.4f,  -4.5f},
                                      {-0.7f,   0.9f,  -1.2f,   1.3f,  -0.7f,  -0.7f,   0.7f},
                                      { 1.0f,   1.1f,   1.6f,  -1.4f,   1.4f,   1.4f,  -1.2f}}).view({1, 1, 32, 7}));
    return grads;
}

inline
std::vector<torch::Tensor> _collectionOf2dMixtures_with4Gs() {
    std::vector<torch::Tensor> mixtures;
    mixtures.emplace_back(torch::tensor({{0.53f,  0.0f,  0.0f,  2.0f,  0.0f,  0.0f,  2.0f},
                                         {0.58f,  0.0f,  0.0f,  2.0f,  0.0f,  0.0f,  2.0f},
                                         {0.52f, 10.0f, 10.0f,  1.0f,  0.0f,  0.0f,  1.0f},
                                         {0.55f, 10.0f, 10.0f,  1.0f,  0.0f,  0.0f,  1.0f}}).view({1, 1, 4, 7}));

    mixtures.emplace_back(torch::tensor({{0.55f,  5.0f,  5.0f,  4.0f, -0.5f, -0.5f,  4.0f},
                                         {0.53f,  8.0f,  8.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.57f, 20.0f, 10.0f,  5.0f,  0.0f,  0.0f,  7.0f},
                                         {0.54f, 20.0f, 20.0f,  5.0f,  0.5f,  0.5f,  7.0f}}).view({1, 1, 4, 7}));

    mixtures.emplace_back(torch::tensor({{1.02f, -0.5f, -0.5f,  4.0f, -1.0f, -1.0f,  4.0f},
                                         {0.53f,  0.5f,  0.5f,  4.0f, -0.8f, -0.8f,  4.0f},
                                         {0.57f, -0.5f,  0.5f,  5.0f,  0.5f,  0.5f,  7.0f},
                                         {1.54f,  0.5f, -0.5f,  5.0f,  0.5f,  0.5f,  7.0f}}).view({1, 1, 4, 7}));

    mixtures.emplace_back(torch::tensor({{1.02f, -2.5f,  2.5f,  4.0f, -1.0f, -1.0f,  4.0f},
                                         {0.53f,  1.8f,  1.8f,  4.0f,  0.8f,  0.8f,  4.0f},
                                         {0.55f, -0.5f,  4.5f,  5.0f,  0.5f,  0.5f,  7.0f},
                                         {1.58f,  0.5f, -4.5f,  5.0f,  0.5f,  0.5f,  7.0f}}).view({1, 1, 4, 7}));

    mixtures.emplace_back(torch::tensor({{0.00f, -2.5f,  2.5f,  4.0f, -1.0f, -1.0f,  4.0f},
                                         {0.54f,  1.8f,  1.8f,  4.0f,  0.8f,  0.8f,  4.0f},
                                         {0.56f, -0.5f,  4.5f,  5.0f,  0.5f,  0.5f,  7.0f},
                                         {1.58f,  0.5f, -4.5f,  5.0f,  0.5f,  0.5f,  7.0f}}).view({1, 1, 4, 7}));

    return mixtures;
}

inline
std::vector<torch::Tensor> _collectionOf2dMixtures_with8GsForQuickAutoDiff() {
    std::vector<torch::Tensor> mixtures;
    mixtures.emplace_back(torch::tensor({{0.0f,  5.0f,  5.0f,  4.0f, -0.5f, -0.5f,  4.0f},
                                         {0.0f,  4.5f,  5.1f,  3.8f, -0.6f, -0.6f,  4.1f},
                                         {0.0f,  8.0f,  8.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.0f,  7.8f,  8.5f,  4.5f, -1.5f, -1.5f,  4.2f},
                                         {0.0f, 18.0f, 11.0f,  6.0f,  0.0f,  0.0f,  8.0f},
                                         {0.0f, 20.0f, 13.0f,  5.4f,  0.0f,  0.0f,  7.0f},
                                         {0.0f, 21.0f, 16.0f,  5.7f,  0.5f,  0.5f,  6.0f},
                                         {0.0f, 19.0f, 17.0f,  4.9f,  0.5f,  0.5f,  5.0f}}).view({1, 1, 8, 7}));

    mixtures.emplace_back(torch::tensor({{0.5f,  5.0f,  5.0f,  4.0f, -0.5f, -0.5f,  4.0f},
                                         {0.0f,  4.5f,  5.1f,  3.8f, -0.6f, -0.6f,  4.1f},
                                         {0.0f,  8.0f,  8.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.0f,  7.8f,  8.5f,  4.5f, -1.5f, -1.5f,  4.2f},
                                         {0.0f, 18.0f, 11.0f,  6.0f,  0.0f,  0.0f,  8.0f},
                                         {0.0f, 20.0f, 13.0f,  5.4f,  0.0f,  0.0f,  7.0f},
                                         {0.0f, 21.0f, 16.0f,  5.7f,  0.5f,  0.5f,  6.0f},
                                         {0.0f, 19.0f, 17.0f,  4.9f,  0.5f,  0.5f,  5.0f}}).view({1, 1, 8, 7}));

    mixtures.emplace_back(torch::tensor({{0.0f,  5.0f,  5.0f,  4.0f, -0.5f, -0.5f,  4.0f},
                                         {0.0f,  4.5f,  5.1f,  3.8f, -0.6f, -0.6f,  4.1f},
                                         {0.5f,  8.0f,  8.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.0f,  7.8f,  8.5f,  4.5f, -1.5f, -1.5f,  4.2f},
                                         {0.0f, 18.0f, 11.0f,  6.0f,  0.0f,  0.0f,  8.0f},
                                         {0.8f, 20.0f, 13.0f,  5.4f,  0.0f,  0.0f,  7.0f},
                                         {0.0f, 21.0f, 16.0f,  5.7f,  0.5f,  0.5f,  6.0f},
                                         {0.0f, 19.0f, 17.0f,  4.9f,  0.5f,  0.5f,  5.0f}}).view({1, 1, 8, 7}));

    mixtures.emplace_back(torch::tensor({{0.0f,  5.0f,  5.0f,  4.0f, -0.5f, -0.5f,  4.0f},
                                         {0.5f,  4.5f,  5.1f,  3.8f, -0.6f, -0.6f,  4.1f},
                                         {0.0f,  8.0f,  8.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.0f,  7.8f,  8.5f,  4.5f, -1.5f, -1.5f,  4.2f},
                                         {0.0f, 18.0f, 11.0f,  6.0f,  0.0f,  0.0f,  8.0f},
                                         {0.0f, 20.0f, 13.0f,  5.4f,  0.0f,  0.0f,  7.0f},
                                         {0.8f, 21.0f, 16.0f,  5.7f,  0.5f,  0.5f,  6.0f},
                                         {0.0f, 19.0f, 17.0f,  4.9f,  0.5f,  0.5f,  5.0f}}).view({1, 1, 8, 7}));

    return mixtures;
}
inline
std::vector<torch::Tensor> _collectionOf2dMixtures_with8GsForMediumLongAutoDiff() {
    std::vector<torch::Tensor> mixtures;

    mixtures.emplace_back(torch::tensor({{0.00f,  5.0f,  5.0f,  4.0f, -0.5f, -0.5f,  4.0f},
                                         {0.49f,  4.5f,  5.1f,  3.8f, -0.6f, -0.6f,  4.1f},
                                         {0.00f,  8.0f,  8.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.68f,  7.8f,  8.5f,  4.5f, -1.5f, -1.5f,  4.2f},
                                         {0.00f, 18.0f, 11.0f,  6.0f,  0.0f,  0.0f,  8.0f},
                                         {0.86f, 20.0f, 13.0f,  5.4f,  0.0f,  0.0f,  7.0f},
                                         {0.00f, 21.0f, 16.0f,  5.7f,  0.5f,  0.5f,  6.0f},
                                         {1.04f, 19.0f, 17.0f,  4.9f,  0.5f,  0.5f,  5.0f}}).view({1, 1, 8, 7}));

    mixtures.emplace_back(torch::tensor({{0.33f,  5.0f,  5.0f,  4.0f, -0.5f, -0.5f,  4.0f},
                                         {0.00f,  4.5f,  5.1f,  3.8f, -0.6f, -0.6f,  4.1f},
                                         {0.00f,  8.0f,  8.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.00f,  7.8f,  8.5f,  4.5f, -1.5f, -1.5f,  4.2f},
                                         {0.78f, 18.0f, 11.0f,  6.0f,  0.0f,  0.0f,  8.0f},
                                         {0.85f, 20.0f, 13.0f,  5.4f,  0.0f,  0.0f,  7.0f},
                                         {0.00f, 21.0f, 16.0f,  5.7f,  0.5f,  0.5f,  6.0f},
                                         {1.02f, 19.0f, 17.0f,  4.9f,  0.5f,  0.5f,  5.0f}}).view({1, 1, 8, 7}));

    mixtures.emplace_back(torch::tensor({{0.37f,  5.0f,  5.0f,  4.0f, -0.5f, -0.5f,  4.0f},
                                         {0.00f,  4.5f,  5.1f,  3.8f, -0.6f, -0.6f,  4.1f},
                                         {0.53f,  8.0f,  8.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.00f,  7.8f,  8.5f,  4.5f, -1.5f, -1.5f,  4.2f},
                                         {0.78f, 18.0f, 11.0f,  6.0f,  0.0f,  0.0f,  8.0f},
                                         {0.00f, 20.0f, 13.0f,  5.4f,  0.0f,  0.0f,  7.0f},
                                         {0.92f, 21.0f, 16.0f,  5.7f,  0.5f,  0.5f,  6.0f},
                                         {0.00f, 19.0f, 17.0f,  4.9f,  0.5f,  0.5f,  5.0f}}).view({1, 1, 8, 7}));

    mixtures.emplace_back(torch::tensor({{0.00f,  5.0f,  5.0f,  4.0f, -0.5f, -0.5f,  4.0f},
                                         {0.00f,  4.5f,  5.1f,  3.8f, -0.6f, -0.6f,  4.1f},
                                         {0.56f,  8.0f,  8.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.00f,  7.8f,  8.5f,  4.5f, -1.5f, -1.5f,  4.2f},
                                         {0.00f, 18.0f, 11.0f,  6.0f,  0.0f,  0.0f,  8.0f},
                                         {0.84f, 20.0f, 13.0f,  5.4f,  0.0f,  0.0f,  7.0f},
                                         {0.93f, 21.0f, 16.0f,  5.7f,  0.5f,  0.5f,  6.0f},
                                         {1.08f, 19.0f, 17.0f,  4.9f,  0.5f,  0.5f,  5.0f}}).view({1, 1, 8, 7}));

    return mixtures;
}

inline
std::vector<torch::Tensor> _collectionOf2dMixtures_with8GsForLongAutoDiff() {
    std::vector<torch::Tensor> mixtures;
    mixtures.emplace_back(torch::tensor({{0.00f,  5.0f,  5.0f,  4.0f, -0.5f, -0.5f,  4.0f},
                                         {0.42f,  4.5f,  5.1f,  3.8f, -0.6f, -0.6f,  4.1f},
                                         {0.58f,  8.0f,  8.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.63f,  7.8f,  8.5f,  4.5f, -1.5f, -1.5f,  4.2f},
                                         {0.00f, 18.0f, 11.0f,  6.0f,  0.0f,  0.0f,  8.0f},
                                         {0.86f, 20.0f, 13.0f,  5.4f,  0.0f,  0.0f,  7.0f},
                                         {0.00f, 21.0f, 16.0f,  5.7f,  0.5f,  0.5f,  6.0f},
                                         {1.09f, 19.0f, 17.0f,  4.9f,  0.5f,  0.5f,  5.0f}}).view({1, 1, 8, 7}));

    mixtures.emplace_back(torch::tensor({{0.52f,  0.0f,  0.0f,  2.0f,  0.0f,  0.0f,  2.0f},
                                         {0.00f,  0.0f,  0.0f,  2.0f,  0.0f,  0.0f,  2.0f},
                                         {0.52f,  0.0f,  0.0f,  2.0f,  0.0f,  0.0f,  2.0f},
                                         {0.00f,  0.0f,  0.0f,  2.0f,  0.0f,  0.0f,  2.0f},
                                         {0.54f, 10.0f, 10.0f,  1.0f,  0.0f,  0.0f,  1.0f},
                                         {0.58f, 10.0f, 10.0f,  1.0f,  0.0f,  0.0f,  1.0f},
                                         {0.00f, 10.0f, 10.0f,  1.0f,  0.0f,  0.0f,  1.0f},
                                         {0.53f, 10.0f, 10.0f,  1.0f,  0.0f,  0.0f,  1.0f}}).view({1, 1, 8, 7}));

    mixtures.emplace_back(torch::tensor({{1.03f,  0.0f,  0.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.00f,  5.0f,  0.0f,  2.0f, -1.5f, -1.5f,  3.0f},
                                         {0.58f, 10.0f, 10.0f,  4.0f,  0.5f,  0.5f,  4.0f},
                                         {1.55f, 15.0f, 15.0f,  5.0f, -0.5f, -0.5f,  4.0f},
                                         {0.00f, 20.0f, 20.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.58f, 25.0f, 25.0f,  2.0f, -1.5f, -1.5f,  3.0f},
                                         {0.00f, 30.0f, 35.0f,  4.0f,  0.5f,  0.5f,  4.0f},
                                         {1.54f, 35.0f, 30.0f,  5.0f, -0.5f, -0.5f,  4.0f}}).view({1, 1, 8, 7}));

    mixtures.emplace_back(torch::tensor({{1.04f, -0.5f, -0.5f,  4.0f, -1.0f, -1.0f,  4.0f},
                                         {0.00f,  0.5f,  0.5f,  4.0f, -0.8f, -0.8f,  4.0f},
                                         {0.74f, -0.5f,  0.5f,  5.0f,  0.5f,  0.5f,  7.0f},
                                         {1.18f,  0.5f, -0.5f,  5.0f,  0.5f,  0.5f,  7.0f},
                                         {1.23f, -0.7f, -0.3f,  3.0f, -1.0f, -1.0f,  4.0f},
                                         {0.95f,  0.3f,  0.7f,  4.0f, -0.8f, -0.8f,  5.0f},
                                         {0.00f, -0.7f,  0.3f,  5.0f,  0.5f,  0.5f,  6.0f},
                                         {0.00f,  0.3f, -0.7f,  3.0f,  0.5f,  0.5f,  7.0f}}).view({1, 1, 8, 7}));

    mixtures.emplace_back(torch::tensor({{0.00f, -2.5f,  2.5f,  4.0f, -1.0f, -1.0f,  4.0f},
                                         {0.00f,  1.8f,  1.8f,  4.0f,  0.8f,  0.8f,  4.0f},
                                         {0.75f, -0.5f,  4.5f,  5.0f,  0.5f,  0.5f,  7.0f},
                                         {1.13f,  0.5f, -4.5f,  5.0f,  0.5f,  0.5f,  7.0f},
                                         {1.24f, -1.5f,  0.5f,  4.0f, -1.0f, -1.0f,  4.0f},
                                         {0.96f,  0.8f,  0.8f,  4.0f,  0.8f,  0.8f,  4.0f},
                                         {0.00f, -0.5f,  7.5f,  5.0f,  0.5f,  0.5f,  7.0f},
                                         {1.32f,  0.5f, -7.5f,  5.0f,  0.5f,  0.5f,  7.0f}}).view({1, 1, 8, 7}));
    return mixtures;
}

inline
std::vector<torch::Tensor> _collectionOf2dMixtures_causingNumericalProblems() {
    std::vector<torch::Tensor> mixtures;

    mixtures.emplace_back(torch::tensor({{1.03f,   0.0f,   0.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.94f,  10.0f,  10.0f,  2.0f, -1.5f, -1.5f,  3.0f},
                                         {0.59f,  20.0f,  20.0f,  4.0f,  0.5f,  0.5f,  4.0f},
                                         {1.56f,  40.0f,  40.0f,  5.0f, -0.5f, -0.5f,  4.0f},
                                         {0.94f,  60.0f,  60.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.57f,  80.0f,  80.0f,  2.0f, -1.5f, -1.5f,  3.0f},
                                         {0.92f, 120.0f, 120.0f,  4.0f,  0.5f,  0.5f,  4.0f},
                                         {1.58f, 140.0f, 140.0f,  5.0f, -0.5f, -0.5f,  4.0f}}).view({1, 1, 8, 7}));

    mixtures.emplace_back(torch::tensor({{1.04f,  0.0f,  0.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.00f,  5.0f,  5.0f,  2.0f, -1.5f, -1.5f,  3.0f},
                                         {0.59f, 10.0f, 10.0f,  4.0f,  0.5f,  0.5f,  4.0f},
                                         {1.56f, 15.0f, 15.0f,  5.0f, -0.5f, -0.5f,  4.0f},
                                         {0.00f, 20.0f, 20.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.54f, 25.0f, 25.0f,  2.0f, -1.5f, -1.5f,  3.0f},
                                         {0.09f, 30.0f, 35.0f,  4.0f,  0.5f,  0.5f,  4.0f},
                                         {1.56f, 35.0f, 30.0f,  5.0f, -0.5f, -0.5f,  4.0f}}).view({1, 1, 8, 7}));

    mixtures.emplace_back(torch::tensor({{1.06f,  0.0f,  0.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.83f,  5.0f,  5.0f,  2.0f, -1.5f, -1.5f,  3.0f},
                                         {0.55f, 10.0f, 10.0f,  4.0f,  0.5f,  0.5f,  4.0f},
                                         {1.50f, 15.0f, 15.0f,  5.0f, -0.5f, -0.5f,  4.0f},
                                         {1.17f, 20.0f, 20.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.53f, 25.0f, 25.0f,  2.0f, -1.5f, -1.5f,  3.0f},
                                         {1.28f, 30.0f, 35.0f,  4.0f,  0.5f,  0.5f,  4.0f},
                                         {1.56f, 35.0f, 30.0f,  5.0f, -0.5f, -0.5f,  4.0f}}).view({1, 1, 8, 7}));

    mixtures.emplace_back(torch::tensor({{1.06f,   0.0f,   0.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.93f, -10.0f, -10.0f,  2.0f, -1.5f, -1.5f,  3.0f},
                                         {0.56f, -20.0f, -20.0f,  4.0f,  0.5f,  0.5f,  4.0f},
                                         {1.59f, -40.0f, -40.0f,  5.0f, -0.5f, -0.5f,  4.0f},
                                         {0.93f, -60.0f, -60.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.59f, -80.0f, -80.0f,  2.0f, -1.5f, -1.5f,  3.0f},
                                         {0.93f,-120.0f,-120.0f,  4.0f,  0.5f,  0.5f,  4.0f},
                                         {1.56f,-140.0f,-140.0f,  5.0f, -0.5f, -0.5f,  4.0f},
                                         {1.08f,   0.0f,   0.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.94f,  10.0f,  10.0f,  2.0f, -1.5f, -1.5f,  3.0f},
                                         {0.53f,  20.0f,  20.0f,  4.0f,  0.5f,  0.5f,  4.0f},
                                         {1.50f,  40.0f,  40.0f,  5.0f, -0.5f, -0.5f,  4.0f},
                                         {0.98f,  60.0f,  60.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.55f,  80.0f,  80.0f,  2.0f, -1.5f, -1.5f,  3.0f},
                                         {0.93f, 120.0f, 120.0f,  4.0f,  0.5f,  0.5f,  4.0f},
                                         {1.58f, 140.0f, 140.0f,  5.0f, -0.5f, -0.5f,  4.0f}}).view({1, 1, 16, 7}));

    return mixtures;
}

inline
std::vector<torch::Tensor> _collectionOf2dMixtures_with8GsTooLongForAutodiff() {
    std::vector<torch::Tensor> mixtures;
    mixtures.emplace_back(torch::tensor({{0.39f,  5.0f,  5.0f,  4.0f, -0.5f, -0.5f,  4.0f},
                                         {0.44f,  4.5f,  5.1f,  3.8f, -0.6f, -0.6f,  4.1f},
                                         {0.58f,  8.0f,  8.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.65f,  7.8f,  8.5f,  4.5f, -1.5f, -1.5f,  4.2f},
                                         {0.72f, 18.0f, 11.0f,  6.0f,  0.0f,  0.0f,  8.0f},
                                         {0.84f, 20.0f, 13.0f,  5.4f,  0.0f,  0.0f,  7.0f},
                                         {0.95f, 21.0f, 16.0f,  5.7f,  0.5f,  0.5f,  6.0f},
                                         {1.06f, 19.0f, 17.0f,  4.9f,  0.5f,  0.5f,  5.0f}}).view({1, 1, 8, 7}));

    mixtures.emplace_back(torch::tensor({{0.0f,  5.0f,  5.0f,  4.0f, -0.5f, -0.5f,  4.0f},
                                         {0.43f,  4.5f,  5.1f,  3.8f, -0.6f, -0.6f,  4.1f},
                                         {0.55f,  8.0f,  8.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.69f,  7.8f,  8.5f,  4.5f, -1.5f, -1.5f,  4.2f},
                                         {0.0f, 18.0f, 11.0f,  6.0f,  0.0f,  0.0f,  8.0f},
                                         {0.86f, 20.0f, 13.0f,  5.4f,  0.0f,  0.0f,  7.0f},
                                         {0.0f, 21.0f, 16.0f,  5.7f,  0.5f,  0.5f,  6.0f},
                                         {1.04f, 19.0f, 17.0f,  4.9f,  0.5f,  0.5f,  5.0f}}).view({1, 1, 8, 7}));

    mixtures.emplace_back(torch::tensor({{0.56f,  0.0f,  0.0f,  2.0f,  0.0f,  0.0f,  2.0f},
                                         {0.0f,  0.0f,  0.0f,  2.0f,  0.0f,  0.0f,  2.0f},
                                         {0.54f,  0.0f,  0.0f,  2.0f,  0.0f,  0.0f,  2.0f},
                                         {0.0f,  0.0f,  0.0f,  2.0f,  0.0f,  0.0f,  2.0f},
                                         {0.53f, 10.0f, 10.0f,  1.0f,  0.0f,  0.0f,  1.0f},
                                         {0.57f, 10.0f, 10.0f,  1.0f,  0.0f,  0.0f,  1.0f},
                                         {0.0f, 10.0f, 10.0f,  1.0f,  0.0f,  0.0f,  1.0f},
                                         {0.59f, 10.0f, 10.0f,  1.0f,  0.0f,  0.0f,  1.0f}}).view({1, 1, 8, 7}));

    mixtures.emplace_back(torch::tensor({{1.03f,  0.0f,  0.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.0f,  2.0f,  2.0f,  2.0f, -1.5f, -1.5f,  3.0f},
                                         {0.55f,  4.0f,  4.0f,  4.0f,  0.5f,  0.5f,  4.0f},
                                         {1.57f,  6.0f,  6.0f,  5.0f, -0.5f, -0.5f,  4.0f},
                                         {0.0f,  8.0f,  8.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.58f, 10.0f, 10.0f,  2.0f, -1.5f, -1.5f,  3.0f},
                                         {0.0f, 12.0f, 12.0f,  4.0f,  0.5f,  0.5f,  4.0f},
                                         {1.54f, 14.0f, 14.0f,  5.0f, -0.5f, -0.5f,  4.0f}}).view({1, 1, 8, 7}));

    mixtures.emplace_back(torch::tensor({{1.03f,  0.0f,  0.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.89f,  2.0f,  2.0f,  2.0f, -1.5f, -1.5f,  3.0f},
                                         {0.57f,  4.0f,  4.0f,  4.0f,  0.5f,  0.5f,  4.0f},
                                         {1.53f,  6.0f,  6.0f,  5.0f, -0.5f, -0.5f,  4.0f},
                                         {1.15f,  8.0f,  8.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.59f, 10.0f, 10.0f,  2.0f, -1.5f, -1.5f,  3.0f},
                                         {1.26f, 12.0f, 12.0f,  4.0f,  0.5f,  0.5f,  4.0f},
                                         {1.54f, 14.0f, 14.0f,  5.0f, -0.5f, -0.5f,  4.0f}}).view({1, 1, 8, 7}));

    mixtures.emplace_back(torch::tensor({{1.03f, -0.5f, -0.5f,  4.0f, -1.0f, -1.0f,  4.0f},
                                         {0.0f,  0.5f,  0.5f,  4.0f, -0.8f, -0.8f,  4.0f},
                                         {0.78f, -0.5f,  0.5f,  5.0f,  0.5f,  0.5f,  7.0f},
                                         {1.13f,  0.5f, -0.5f,  5.0f,  0.5f,  0.5f,  7.0f},
                                         {1.23f, -0.7f, -0.3f,  3.0f, -1.0f, -1.0f,  4.0f},
                                         {0.94f,  0.3f,  0.7f,  4.0f, -0.8f, -0.8f,  5.0f},
                                         {0.0f, -0.7f,  0.3f,  5.0f,  0.5f,  0.5f,  6.0f},
                                         {0.0f,  0.3f, -0.7f,  3.0f,  0.5f,  0.5f,  7.0f}}).view({1, 1, 8, 7}));

    mixtures.emplace_back(torch::tensor({{1.09f, -0.5f, -0.5f,  4.0f, -1.0f, -1.0f,  4.0f},
                                         {0.88f,  0.5f,  0.5f,  4.0f, -0.8f, -0.8f,  4.0f},
                                         {0.75f, -0.5f,  0.5f,  5.0f,  0.5f,  0.5f,  7.0f},
                                         {1.12f,  0.5f, -0.5f,  5.0f,  0.5f,  0.5f,  7.0f},
                                         {1.25f, -0.7f, -0.3f,  3.0f, -1.0f, -1.0f,  4.0f},
                                         {0.99f,  0.3f,  0.7f,  4.0f, -0.8f, -0.8f,  5.0f},
                                         {1.03f, -0.7f,  0.3f,  5.0f,  0.5f,  0.5f,  6.0f},
                                         {1.34f,  0.3f, -0.7f,  3.0f,  0.5f,  0.5f,  7.0f}}).view({1, 1, 8, 7}));

    mixtures.emplace_back(torch::tensor({{0.00f, -2.5f,  2.5f,  4.0f, -1.0f, -1.0f,  4.0f},
                                         {0.00f,  1.8f,  1.8f,  4.0f,  0.8f,  0.8f,  4.0f},
                                         {0.73f, -0.5f,  4.5f,  5.0f,  0.5f,  0.5f,  7.0f},
                                         {1.18f,  0.5f, -4.5f,  5.0f,  0.5f,  0.5f,  7.0f},
                                         {1.23f, -1.5f,  0.5f,  4.0f, -1.0f, -1.0f,  4.0f},
                                         {0.95f,  0.8f,  0.8f,  4.0f,  0.8f,  0.8f,  4.0f},
                                         {0.00f, -0.5f,  7.5f,  5.0f,  0.5f,  0.5f,  7.0f},
                                         {1.36f,  0.5f, -7.5f,  5.0f,  0.5f,  0.5f,  7.0f}}).view({1, 1, 8, 7}));
    return mixtures;
}

inline
std::vector<torch::Tensor> _collectionOf2dMixtures_with16Gs() {
    std::vector<torch::Tensor> mixtures;
    mixtures.emplace_back(torch::tensor({{0.34f,  5.0f,  5.0f,  4.0f, -0.5f, -0.5f,  4.0f},
                                         {0.45f,  4.5f,  5.1f,  3.8f, -0.6f, -0.6f,  4.1f},
                                         {0.57f,  8.0f,  8.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.63f,  7.8f,  8.5f,  4.5f, -1.5f, -1.5f,  4.2f},
                                         {0.79f, 18.0f, 11.0f,  6.0f,  0.0f,  0.0f,  8.0f},
                                         {0.86f, 20.0f, 13.0f,  5.4f,  0.0f,  0.0f,  7.0f},
                                         {0.95f, 21.0f, 16.0f,  5.7f,  0.5f,  0.5f,  6.0f},
                                         {1.09f, 19.0f, 17.0f,  4.9f,  0.5f,  0.5f,  5.0f},
                                         {1.06f, -0.5f, -0.5f,  4.0f, -1.0f, -1.0f,  4.0f},
                                         {0.83f,  0.5f,  0.5f,  4.0f, -0.8f, -0.8f,  4.0f},
                                         {0.75f, -0.5f,  0.5f,  5.0f,  0.5f,  0.5f,  7.0f},
                                         {1.19f,  0.5f, -0.5f,  5.0f,  0.5f,  0.5f,  7.0f},
                                         {1.22f, -0.7f, -0.3f,  3.0f, -1.0f, -1.0f,  4.0f},
                                         {0.96f,  0.3f,  0.7f,  4.0f, -0.8f, -0.8f,  5.0f},
                                         {1.09f, -0.7f,  0.3f,  5.0f,  0.5f,  0.5f,  6.0f},
                                         {1.30f,  0.3f, -0.7f,  3.0f,  0.5f,  0.5f,  7.0f}}).view({1, 1, 16, 7}));

    mixtures.emplace_back(torch::tensor({{0.0f,  5.0f,  5.0f,  4.0f, -0.5f, -0.5f,  4.0f},
                                         {0.43f,  4.5f,  5.1f,  3.8f, -0.6f, -0.6f,  4.1f},
                                         {0.0f,  8.0f,  8.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.0f,  7.8f,  8.5f,  4.5f, -1.5f, -1.5f,  4.2f},
                                         {0.79f, 18.0f, 11.0f,  6.0f,  0.0f,  0.0f,  8.0f},
                                         {0.0f, 20.0f, 13.0f,  5.4f,  0.0f,  0.0f,  7.0f},
                                         {0.92f, 21.0f, 16.0f,  5.7f,  0.5f,  0.5f,  6.0f},
                                         {1.04f, 19.0f, 17.0f,  4.9f,  0.5f,  0.5f,  5.0f},
                                         {0.0f, -0.5f, -0.5f,  4.0f, -1.0f, -1.0f,  4.0f},
                                         {0.0f,  0.5f,  0.5f,  4.0f, -0.8f, -0.8f,  4.0f},
                                         {0.0f, -0.5f,  0.5f,  5.0f,  0.5f,  0.5f,  7.0f},
                                         {1.19f,  0.5f, -0.5f,  5.0f,  0.5f,  0.5f,  7.0f},
                                         {0.0f, -0.7f, -0.3f,  3.0f, -1.0f, -1.0f,  4.0f},
                                         {0.97f,  0.3f,  0.7f,  4.0f, -0.8f, -0.8f,  5.0f},
                                         {0.0f, -0.7f,  0.3f,  5.0f,  0.5f,  0.5f,  6.0f},
                                         {1.32f,  0.3f, -0.7f,  3.0f,  0.5f,  0.5f,  7.0f}}).view({1, 1, 16, 7}));

    mixtures.emplace_back(torch::tensor({{0.53f,  0.1f,  0.0f,  2.0f,  0.0f,  0.0f,  2.0f},
                                         {0.00f,  0.0f,  0.0f,  2.0f,  0.0f,  0.0f,  2.0f},
                                         {0.57f,  0.0f,  0.0f,  2.0f,  0.0f,  0.0f,  2.0f},
                                         {0.50f, 10.1f, 10.0f,  1.0f,  0.0f,  0.0f,  1.0f},
                                         {0.00f,  0.0f,  0.0f,  2.0f,  0.0f,  0.0f,  2.0f},
                                         {0.00f, 10.0f, 10.0f,  1.0f,  0.0f,  0.0f,  1.0f},
                                         {0.53f, 10.0f, 10.0f,  1.0f,  0.0f,  0.0f,  1.0f},
                                         {0.55f,  0.0f,  0.0f,  2.0f,  0.0f,  0.0f,  2.0f},
                                         {0.58f, 10.0f, 10.0f,  1.0f,  0.0f,  0.0f,  1.0f},
                                         {0.00f,  0.0f,  0.0f,  2.0f,  0.0f,  0.0f,  2.0f},
                                         {0.53f, 10.0f, 10.0f,  1.0f,  0.0f,  0.0f,  1.0f},
                                         {0.56f,  0.0f,  0.0f,  2.0f,  0.0f,  0.0f,  2.0f},
                                         {0.00f,  0.0f,  0.0f,  2.0f,  0.0f,  0.0f,  2.0f},
                                         {0.59f, 10.0f, 10.0f,  1.0f,  0.0f,  0.0f,  1.0f},
                                         {0.00f, 10.0f, 10.0f,  1.0f,  0.0f,  0.0f,  1.0f},
                                         {0.52f, 10.0f, 10.0f,  1.0f,  0.0f,  0.0f,  1.0f}}).view({1, 1, 16, 7}));

    mixtures.emplace_back(torch::tensor({{1.09f,   0.0f, -0.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.83f,  -2.0f, -2.0f,  2.0f, -1.5f, -1.5f,  3.0f},
                                         {0.52f,  -4.0f, -4.0f,  4.0f,  0.5f,  0.5f,  4.0f},
                                         {1.56f,  -6.0f, -6.0f,  5.0f, -0.5f, -0.5f,  4.0f},
                                         {1.19f,  -8.0f, -8.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.54f, -10.0f,-10.0f,  2.0f, -1.5f, -1.5f,  3.0f},
                                         {1.26f, -12.0f,-12.0f,  4.0f,  0.5f,  0.5f,  4.0f},
                                         {1.53f, -14.0f,-14.0f,  5.0f, -0.5f, -0.5f,  4.0f},
                                         {1.08f,   0.0f,  0.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.86f,   2.0f,  2.0f,  2.0f, -1.5f, -1.5f,  3.0f},
                                         {0.53f,   4.0f,  4.0f,  4.0f,  0.5f,  0.5f,  4.0f},
                                         {1.58f,   6.0f,  6.0f,  5.0f, -0.5f, -0.5f,  4.0f},
                                         {1.13f,   8.0f,  8.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.55f,  10.0f, 10.0f,  2.0f, -1.5f, -1.5f,  3.0f},
                                         {1.29f,  12.0f, 12.0f,  4.0f,  0.5f,  0.5f,  4.0f},
                                         {1.54f,  14.0f, 14.0f,  5.0f, -0.5f, -0.5f,  4.0f}}).view({1, 1, 16, 7}));

    mixtures.emplace_back(torch::tensor({{1.08f,  1.0f,  1.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.83f, -2.0f,  2.0f,  2.0f, -1.5f, -1.5f,  3.0f},
                                         {1.53f, 14.0f,-14.0f,  5.0f, -0.5f, -0.5f,  4.0f},
                                         {0.58f, 10.0f,-10.0f,  2.0f, -1.5f, -1.5f,  3.0f},
                                         {0.54f,-10.0f, 10.0f,  2.0f, -1.5f, -1.5f,  3.0f},
                                         {1.12f, -8.0f,  8.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {1.04f,  0.0f,  0.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {1.58f,  6.0f, -6.0f,  5.0f, -0.5f, -0.5f,  4.0f},
                                         {1.22f,-12.0f, 12.0f,  4.0f,  0.5f,  0.5f,  4.0f},
                                         {0.58f,  4.0f, -4.0f,  4.0f,  0.5f,  0.5f,  4.0f},
                                         {1.56f, -6.0f,  6.0f,  5.0f, -0.5f, -0.5f,  4.0f},
                                         {1.22f, 12.0f,-12.0f,  4.0f,  0.5f,  0.5f,  4.0f},
                                         {0.88f,  2.0f, -2.0f,  2.0f, -1.5f, -1.5f,  3.0f},
                                         {1.12f,  8.0f, -8.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {1.59f,-14.0f, 14.0f,  5.0f, -0.5f, -0.5f,  4.0f},
                                         {0.54f, -4.0f,  4.0f,  4.0f,  0.5f,  0.5f,  4.0f}}).view({1, 1, 16, 7}));

    mixtures.emplace_back(torch::tensor({{1.04f, -1.0f,  1.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.86f,  2.0f,  2.0f,  2.0f, -1.5f, -1.5f,  3.0f},
                                         {1.59f, 14.0f, 14.0f,  5.0f, -0.5f, -0.5f,  4.0f},
                                         {0.55f,-10.0f,-10.0f,  2.0f, -1.5f, -1.5f,  3.0f},
                                         {0.53f, 10.0f,  5.0f,  2.0f, -1.5f, -1.5f,  3.0f},
                                         {1.17f, -8.0f,  2.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {1.09f,  2.0f,  0.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {1.56f,  0.0f, -6.0f,  5.0f, -0.5f, -0.5f,  4.0f},
                                         {1.23f,-12.0f,  8.0f,  4.0f,  0.5f,  0.5f,  4.0f},
                                         {0.55f,  6.0f, -4.0f,  4.0f,  0.5f,  0.5f,  4.0f},
                                         {1.59f,  6.0f,  6.0f,  5.0f, -0.5f, -0.5f,  4.0f},
                                         {1.23f,  5.0f,-12.0f,  4.0f,  0.5f,  0.5f,  4.0f},
                                         {0.86f,  2.0f,  3.0f,  2.0f, -1.5f, -1.5f,  3.0f},
                                         {1.19f,  8.0f,  5.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {1.54f, 12.0f,  9.0f,  5.0f, -0.5f, -0.5f,  4.0f},
                                         {0.56f, -4.0f,  4.0f,  4.0f,  0.5f,  0.5f,  4.0f}}).view({1, 1, 16, 7}));
    return mixtures;
}

inline
std::vector<torch::Tensor> _collectionOf2dMixtures_with128Gs_red2() {
    using namespace torch::indexing;
    std::vector<torch::Tensor> mixtures;

    {
        auto t = torch::rand({128, 7});
        t.index_put_({Slice(), 0}, 0);
        t.index_put_({Slice(), 4}, 0);
        t.index_put_({Slice(), 5}, 0);
        mixtures.emplace_back(t.view({1, 1, 128, 7}));
    }
    {
        auto t = torch::rand({128, 7});
        t.index_put_({Slice(), 0}, 0);
        t.index_put_({Slice(), 4}, 0);
        t.index_put_({Slice(), 5}, 0);
        t.index_put_({0, 0}, 1.5);
        t.index_put_({1, 0}, 0.5);
        t.index_put_({3, 0}, 1.1);
        mixtures.emplace_back(t.view({1, 1, 128, 7}));
    }

    {
        auto t = torch::rand({128, 7});
        t.index_put_({Slice(), 0}, 0);
        t.index_put_({Slice(), 4}, 0);
        t.index_put_({Slice(), 5}, 0);
        for (auto i = 0; i < 8; ++i) {
            for (auto j = 0; j < 16; ++j) {
                t.index_put_({i * 16 + j, 1}, float(i));
                t.index_put_({i * 16 + j, 2}, float(j));

            }
        }
        t.index_put_({0, 0}, 1.0);
        t.index_put_({3*16+14, 0}, 1.4);
        mixtures.emplace_back(t.view({1, 1, 128, 7}));
    }

    return mixtures;
}
inline
std::vector<torch::Tensor> _collectionOf2dMixtures_with128Gs_red4() {
    using namespace torch::indexing;
    std::vector<torch::Tensor> mixtures;

    {
        auto t = torch::rand({128, 7});
        t.index_put_({Slice(), 0}, 0);
        t.index_put_({Slice(), 4}, 0);
        t.index_put_({Slice(), 5}, 0);
        t.index_put_({0, 0}, 1.5);
        t.index_put_({1, 0}, 0.5);
        t.index_put_({2, 0}, 0.8);
        t.index_put_({3, 0}, 1.1);
        mixtures.emplace_back(t.view({1, 1, 128, 7}));
    }
    {
        auto t = torch::rand({128, 7});
        t.index_put_({Slice(), 0}, 0);
        t.index_put_({Slice(), 4}, 0);
        t.index_put_({Slice(), 5}, 0);
        for (auto i = 0; i < 8; ++i) {
            for (auto j = 0; j < 16; ++j) {
                t.index_put_({i * 16 + j, 1}, float(i));
                t.index_put_({i * 16 + j, 2}, float(j));

            }
        }
        t.index_put_({0, 0}, 1.0);
        t.index_put_({3*16+2, 0}, 1.5);
        t.index_put_({5*16+5, 0}, 0.9);
        t.index_put_({16+8, 0}, 1.2);
        t.index_put_({3*16+14, 0}, 1.4);
        mixtures.emplace_back(t.view({1, 1, 128, 7}));
    }

    return mixtures;
}

inline
std::vector<torch::Tensor> _collectionOf2dMixtures_with128Gs_red8() {
    using namespace torch::indexing;
    std::vector<torch::Tensor> mixtures;
    {
        auto t = torch::rand({128, 7});
        t.index_put_({Slice(), 0}, 0);
        t.index_put_({Slice(), 4}, 0);
        t.index_put_({Slice(), 5}, 0);
        for (auto i = 0; i < 8; ++i) {
            for (auto j = 0; j < 16; ++j) {
                t.index_put_({i * 16 + j, 1}, float(i));
                t.index_put_({i * 16 + j, 2}, float(j));

            }
        }
        t.index_put_({0, 0}, 1.0);
        t.index_put_({2, 0}, 0.8);
        t.index_put_({3*16+2, 0}, 1.5);
        t.index_put_({4, 0}, 1.1);
        t.index_put_({5*16+1, 0}, 1.6);
        t.index_put_({5*16+5, 0}, 0.9);
        t.index_put_({16+8, 0}, 1.2);
        t.index_put_({3*16+14, 0}, 1.4);
        mixtures.emplace_back(t.view({1, 1, 128, 7}));
    }
    {
        auto t = torch::rand({128, 7});
        t.index_put_({Slice(), 0}, 0);
        t.index_put_({Slice(), 4}, 0);
        t.index_put_({Slice(), 5}, 0);
        for (auto i = 0; i < 8; ++i) {
            for (auto j = 0; j < 16; ++j) {
                t.index_put_({i * 16 + j, 1}, float(i));
                t.index_put_({i * 16 + j, 2}, float(j));

            }
        }
        t.index_put_({2, 0}, 0.8);
        t.index_put_({3*16+2, 0}, 1.5);
        t.index_put_({5*16+1, 0}, 1.6);
        t.index_put_({5*16+5, 0}, 0.9);
        t.index_put_({16+8, 0}, 1.2);
        t.index_put_({3*16+14, 0}, 1.4);
        mixtures.emplace_back(t.view({1, 1, 128, 7}));
    }

    return mixtures;
}


template <typename T>
std::vector<T> concat(std::initializer_list<std::vector<T>> list) {
    std::vector<T> r;
    for (const auto& v : list) {
        std::copy(v.begin(), v.end(), std::back_inserter(r));
    }
    return r;
}

inline
std::vector<std::pair<torch::Tensor, torch::Tensor>> _combineCollectionsOfGradsAndMixtures(std::initializer_list<std::vector<torch::Tensor>> grads, std::initializer_list<std::vector<torch::Tensor>> mixtures) {
    std::vector<std::pair<torch::Tensor, torch::Tensor>> test_cases;
    for (const auto& grad : concat(grads)) {
        for (const auto& mixture : concat(mixtures)) {
            test_cases.emplace_back(mixture, grad);
        }
    }
    return test_cases;
}

#endif // UNIT_TEST_SUPPORT_H
