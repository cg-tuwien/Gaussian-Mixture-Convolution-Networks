#ifndef NDEBUG
#include <iostream>

#include "util/glm.h"

#include "util/algorithms.h"
#include "util/autodiff.h"
#include "util/mixture.h"
#include "util/gaussian.h"
#include "util/grad/algorithms.h"
#include "util/grad/gaussian.h"
#include "util/grad/scalar.h"
#include "util/grad/glm.h"
#include "util/scalar.h"


namespace  {
using AutodiffScalar = autodiff::Variable<float>;

static struct UnitTests {
    UnitTests() {
        test_determinant<2>();
        test_determinant<3>();
        test_gaussian_amplitude<2>();
        test_gaussian_amplitude<3>();
        test_evaluate<2>();
        test_evaluate<3>();
//        test_likelihood<2>();
//        test_likelihood<3>();
        test_vecOnVec<2>();
        test_vecOnVec<3>();
        test_scalarGrads();
        test_matrix_inverse<2>();
        test_matrix_inverse<3>();

        std::cout << "unit tests for mixture_grad done" << std::endl;
    }

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
        for (auto weight : _scalarCollection()) {
            for (auto pos : _vecCollection<DIMS>()) {
                for (auto cov : _covCollection<DIMS>()) {
                    collection.push_back(G{weight, pos, cov});
                }
            }
        }
        return collection;
    }

    template<int DIMS>
    void test_likelihood() {
        for (auto grad : _scalarCollection()) {
            for (auto g1 : _gaussianCollection<DIMS>()) {
                for (auto g2 : _gaussianCollection<DIMS>()) {
                    test_binarycase(g1, g2, grad, gpe::likelihood<AutodiffScalar, DIMS>, gpe::grad::likelihood<float, DIMS>);
                }
            }
        }
    }

    template<int DIMS>
    void test_evaluate() {
        for (auto grad : _scalarCollection()) {
            for (auto g : _gaussianCollection<DIMS>()) {
                for (auto p : _vecCollection<DIMS>()) {
                    test_binarycase(g, p, grad, gpe::evaluate<AutodiffScalar, DIMS>, gpe::grad::evaluate<float, DIMS>);
                }
            }
        }
    }

    template<int DIMS>
    void test_gaussian_amplitude() {
        for (float grad : _scalarCollection()) {
            for (const auto& cov : _covCollection<DIMS>()) {
                test_unarycase(cov, grad, gpe::gaussian_amplitude<AutodiffScalar, DIMS>, gpe::grad::gaussian_amplitude<float, DIMS>);
            }
        }
    }

    template<int DIMS>
    void test_determinant() {
        for (float grad = -4.f; grad < 4.5f; grad++) {
            test_unarycase(glm::mat<DIMS, DIMS, float>(1), grad, determinant<DIMS>, gpe::grad::determinant<float, DIMS>);
            test_unarycase(glm::mat<DIMS, DIMS, float>(1) * 0.5f, grad, determinant<DIMS>, gpe::grad::determinant<float, DIMS>);
            test_unarycase(glm::mat<DIMS, DIMS, float>(1) * 2.0f, grad, determinant<DIMS>, gpe::grad::determinant<float, DIMS>);
            test_unarycase(glm::mat<DIMS, DIMS, float>(1) * 2.0f + 0.5f, grad, determinant<DIMS>, gpe::grad::determinant<float, DIMS>);
            test_unarycase(glm::mat<DIMS, DIMS, float>(1) * 2.0f - 0.5f, grad, determinant<DIMS>, gpe::grad::determinant<float, DIMS>);

            test_unarycase(_cov<DIMS>(1.5f, -0.1f,  0.2f, 2.5f, -0.3f, 2.2f), grad, determinant<DIMS>, gpe::grad::determinant<float, DIMS>);
            test_unarycase(_cov<DIMS>(2.5f,  0.5f, -0.2f, 1.5f,  0.3f, 3.2f), grad, determinant<DIMS>, gpe::grad::determinant<float, DIMS>);
        }
    }

    template<int DIMS>
    void test_vecOnVec() {
        for (auto a : _vecCollection<DIMS>()) {
            for (auto b : _vecCollection<DIMS>()) {
                for (auto grad : _covCollection<DIMS>()) {
                    test_binarycase(a, b, grad, outerProduct<DIMS>, gpe::grad::outerProduct<DIMS, float>);
                }
                for (auto grad : _scalarCollection()) {
                    test_binarycase(a, b, grad, dot<DIMS>, gpe::grad::dot<DIMS, float>);
                }
            }
        }
    }

    void test_scalarGrads() {
        std::vector<float> values = {0.f, 1.f, -1.f, 1.5, -1.5, 2.3f, -2.5f};
        for (auto a : values) {
            for (auto b : values) {
                for (auto grad : values) {
                    test_scalarGrads_funs(a, b, grad);
                }
            }
        }
    }

    static AutodiffScalar pow(AutodiffScalar a, AutodiffScalar b) { return gpe::pow(a, b); }
    static AutodiffScalar exp(AutodiffScalar a) { return gpe::exp(a); }
    static AutodiffScalar log(AutodiffScalar a) { return gpe::log(a); }
    template<int N_DIMS>
    static glm::mat<N_DIMS, N_DIMS, AutodiffScalar> outerProduct(const glm::vec<N_DIMS, AutodiffScalar>& a, const glm::vec<N_DIMS, AutodiffScalar>& b) { return glm::outerProduct(a, b); }
    template<int N_DIMS>
    static AutodiffScalar determinant(const glm::mat<N_DIMS, N_DIMS, AutodiffScalar>& m) { return glm::determinant(m); }
    template<int N_DIMS>
    static glm::mat<N_DIMS, N_DIMS, AutodiffScalar> matrix_inverse(const glm::mat<N_DIMS, N_DIMS, AutodiffScalar>& m) { return glm::inverse(m); }
    template<int N_DIMS>
    static AutodiffScalar dot(const glm::vec<N_DIMS, AutodiffScalar>& a, const glm::vec<N_DIMS, AutodiffScalar>& b) { return glm::dot(a, b); }

    void test_scalarGrads_funs(float a, float b, float grad) {
//        test_scalarGrads_binarycase(a, b, grad, gpe::functors::times<AutodiffScalar>, gpe::grad::functors::times<float>);
        if (a > 0) {
            test_binarycase(a, b, grad, pow, gpe::grad::pow<float>);
        test_unarycase(a, grad, log, gpe::grad::log<float>);
        }

        test_unarycase(a, grad, exp, gpe::grad::exp<float>);
    }

    template<int N_DIMS>
    void test_matrix_inverse() {
        for (auto grad : _covCollection<N_DIMS>()) {
            for (auto cov : _covCollection<N_DIMS>()) {
                test_unarycase(cov, grad, matrix_inverse<N_DIMS>, gpe::grad::inverse<float, N_DIMS>);
            }
        }
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

} unit_tests;

} // anonymous namespace

#endif // not NDEBUG
