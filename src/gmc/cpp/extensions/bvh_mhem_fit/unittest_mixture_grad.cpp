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
//        test_likelihood<2>();
//        test_likelihood<3>();
        test_outerProduct();
        test_scalarGrads();

        std::cout << "unit tests for mixture_grad done" << std::endl;
    }

    void assert_similar(float a, float b) {
        auto v = std::abs((a + b) / 2);
        v = (v > 0.00001f) ? v : 0.00001f;
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

    template<int DIMS>
    void test_likelihood() {
        for (float grad = -4.f; grad < 4.5f; grad++) {
            test_likelihood_case<DIMS>({0.f, _vec<DIMS>(0, 0, 0), glm::mat<DIMS, DIMS, float>(1)}, {0.f, _vec<DIMS>(0, 0, 0), glm::mat<DIMS, DIMS, float>(1)}, grad);
            test_likelihood_case<DIMS>({0.f, _vec<DIMS>(0, 0, 0), glm::mat<DIMS, DIMS, float>(1)}, {0.f, _vec<DIMS>(1, 1, 1), glm::mat<DIMS, DIMS, float>(1)}, grad);
            test_likelihood_case<DIMS>({0.f, _vec<DIMS>(1, 1, 1), glm::mat<DIMS, DIMS, float>(1)}, {0.f, _vec<DIMS>(0, 0, 0), glm::mat<DIMS, DIMS, float>(1)}, grad);

            test_likelihood_case<DIMS>({0.5f, _vec<DIMS>(0, 0, 0), glm::mat<DIMS, DIMS, float>(1)}, {0.7f, _vec<DIMS>(0, 0, 0), glm::mat<DIMS, DIMS, float>(1)}, grad);
            test_likelihood_case<DIMS>({0.5f, _vec<DIMS>(0, 0, 0), glm::mat<DIMS, DIMS, float>(1)}, {0.7f, _vec<DIMS>(1, 1, 1), glm::mat<DIMS, DIMS, float>(1)}, grad);
            test_likelihood_case<DIMS>({0.5f, _vec<DIMS>(1, 1, 1), glm::mat<DIMS, DIMS, float>(1)}, {0.7f, _vec<DIMS>(0, 0, 0), glm::mat<DIMS, DIMS, float>(1)}, grad);

            test_likelihood_case<DIMS>({0.5f, _vec<DIMS>(1, 2, 3), glm::mat<DIMS, DIMS, float>(1)}, {0.5f, _vec<DIMS>(-1.5, 1.5, 2.5), glm::mat<DIMS, DIMS, float>(1)}, grad);
            test_likelihood_case<DIMS>({1.0f, _vec<DIMS>(1, 2, 3), glm::mat<DIMS, DIMS, float>(1)}, {1.0f, _vec<DIMS>(-1.5, 1.5, 2.5), glm::mat<DIMS, DIMS, float>(1)}, grad);

            test_likelihood_case<DIMS>({0.f, _vec<DIMS>(1, 2, 3), glm::mat<DIMS, DIMS, float>(1) * 0.5f}, {0.f, _vec<DIMS>(-1.5, 1.5, 2.5), glm::mat<DIMS, DIMS, float>(1) * 2.0f + 0.5f}, grad);
            test_likelihood_case<DIMS>({0.5f, _vec<DIMS>(1, 2, 3), glm::mat<DIMS, DIMS, float>(1) * 0.5f}, {0.5f, _vec<DIMS>(-1.5, 1.5, 2.5), glm::mat<DIMS, DIMS, float>(1) * 2.0f + 0.5f}, grad);
            test_likelihood_case<DIMS>({1.0f, _vec<DIMS>(1, 2, 3), glm::mat<DIMS, DIMS, float>(1) * 0.5f}, {1.0f, _vec<DIMS>(-1.5, 1.5, 2.5), glm::mat<DIMS, DIMS, float>(1) * 2.0f + 0.5f}, grad);

            test_likelihood_case<DIMS>({0.f, _vec<DIMS>(1, 2, 3), glm::mat<DIMS, DIMS, float>(1) * 2.0f + 0.5f}, {0.f, _vec<DIMS>(-1.5, 1.5, 2.5), glm::mat<DIMS, DIMS, float>(1) * 0.5f}, grad);
            test_likelihood_case<DIMS>({0.5f, _vec<DIMS>(1, 2, 3), glm::mat<DIMS, DIMS, float>(1) * 2.0f + 0.5f}, {0.5f, _vec<DIMS>(-1.5, 1.5, 2.5), glm::mat<DIMS, DIMS, float>(1) * 0.5f}, grad);
            test_likelihood_case<DIMS>({1.0f, _vec<DIMS>(1, 2, 3), glm::mat<DIMS, DIMS, float>(1) * 2.0f + 0.5f}, {1.0f, _vec<DIMS>(-1.5, 1.5, 2.5), glm::mat<DIMS, DIMS, float>(1) * 0.5f}, grad);

            test_likelihood_case<DIMS>({0.f, _vec<DIMS>(1, 2, 3), glm::mat<DIMS, DIMS, float>(1) * 2.0f + 0.5f}, {0.f, _vec<DIMS>(-1.5, 1.5, 2.5), glm::mat<DIMS, DIMS, float>(1) * 2.0f - 0.5f}, grad);
            test_likelihood_case<DIMS>({0.5f, _vec<DIMS>(1, 2, 3), glm::mat<DIMS, DIMS, float>(1) * 2.0f + 0.5f}, {0.5f, _vec<DIMS>(-1.5, 1.5, 2.5), glm::mat<DIMS, DIMS, float>(1) * 2.0f - 0.5f}, grad);
            test_likelihood_case<DIMS>({1.0f, _vec<DIMS>(1, 2, 3), glm::mat<DIMS, DIMS, float>(1) * 2.0f + 0.5f}, {1.0f, _vec<DIMS>(-1.5, 1.5, 2.5), glm::mat<DIMS, DIMS, float>(1) * 2.0f - 0.5f}, grad);

            test_likelihood_case<DIMS>({0.f, _vec<DIMS>(1, 2, 3), glm::mat<DIMS, DIMS, float>(1) * 2.0f - 0.5f}, {0.f, _vec<DIMS>(-1.5, 1.5, 2.5), glm::mat<DIMS, DIMS, float>(1) * 2.0f + 0.5f}, grad);
            test_likelihood_case<DIMS>({0.5f, _vec<DIMS>(1, 2, 3), glm::mat<DIMS, DIMS, float>(1) * 2.0f - 0.5f}, {0.5f, _vec<DIMS>(-1.5, 1.5, 2.5), glm::mat<DIMS, DIMS, float>(1) * 2.0f + 0.5f}, grad);
            test_likelihood_case<DIMS>({1.0f, _vec<DIMS>(1, 2, 3), glm::mat<DIMS, DIMS, float>(1) * 2.0f - 0.5f}, {1.0f, _vec<DIMS>(-1.5, 1.5, 2.5), glm::mat<DIMS, DIMS, float>(1) * 2.0f + 0.5f}, grad);

            test_likelihood_case<DIMS>({0.f, _vec<DIMS>(1, 2, 3), _cov<DIMS>(1.5f, -0.1f, 0.2f, 2.5f, -0.3f, 2.2f)}, {0.f, _vec<DIMS>(-1.5, 1.5, 2.5), _cov<DIMS>(1.5f, -0.1f, 0.2f, 2.5f, -0.3f, 2.2f)}, grad);
            test_likelihood_case<DIMS>({0.5f, _vec<DIMS>(1, 2, 3), _cov<DIMS>(1.5f, -0.1f, 0.2f, 2.5f, -0.3f, 2.2f)}, {0.5f, _vec<DIMS>(-1.5, 1.5, 2.5), _cov<DIMS>(1.5f, -0.1f, 0.2f, 2.5f, -0.3f, 2.2f)}, grad);
            test_likelihood_case<DIMS>({1.0f, _vec<DIMS>(1, 2, 3), _cov<DIMS>(1.5f, -0.1f, 0.2f, 2.5f, -0.3f, 2.2f)}, {1.0f, _vec<DIMS>(-1.5, 1.5, 2.5), _cov<DIMS>(1.5f, -0.1f, 0.2f, 2.5f, -0.3f, 2.2f)}, grad);

            test_likelihood_case<DIMS>({0.f, _vec<DIMS>(1, 2, 3), _cov<DIMS>(1.5f, -0.1f, 0.2f, 2.5f, -0.3f, 2.2f)}, {0.f, _vec<DIMS>(-1.5, 1.5, 2.5), _cov<DIMS>(2.5f, 0.5f, -0.2f, 1.5f, 0.3f, 3.2f)}, grad);
            test_likelihood_case<DIMS>({0.5f, _vec<DIMS>(1, 2, 3), _cov<DIMS>(1.5f, -0.1f, 0.2f, 2.5f, -0.3f, 2.2f)}, {0.5f, _vec<DIMS>(-1.5, 1.5, 2.5), _cov<DIMS>(2.5f, 0.5f, -0.2f, 1.5f, 0.3f, 3.2f)}, grad);
            test_likelihood_case<DIMS>({1.0f, _vec<DIMS>(1, 2, 3), _cov<DIMS>(1.5f, -0.1f, 0.2f, 2.5f, -0.3f, 2.2f)}, {1.0f, _vec<DIMS>(-1.5, 1.5, 2.5), _cov<DIMS>(2.5f, 0.5f, -0.2f, 1.5f, 0.3f, 3.2f)}, grad);
        }
    }

    template<int DIMS>
    void test_likelihood_case(const gpe::Gaussian<DIMS, float>& t, const gpe::Gaussian<DIMS, float>& f, float grad) {
        test_binarycase(t, f, grad, gpe::likelihood<AutodiffScalar, DIMS>, gpe::grad::likelihood<float, DIMS>);
    }

    template<int DIMS>
    void test_gaussian_amplitude() {
        for (float grad = -4.f; grad < 4.5f; grad++) {
            test_gaussian_amplitude_case(glm::mat<DIMS, DIMS, float>(1), grad);
            test_gaussian_amplitude_case(glm::mat<DIMS, DIMS, float>(1) * 0.5f, grad);
            test_gaussian_amplitude_case(glm::mat<DIMS, DIMS, float>(1) * 2.0f, grad);
            test_gaussian_amplitude_case(glm::mat<DIMS, DIMS, float>(1) * 2.0f + 0.5f, grad);
            test_gaussian_amplitude_case(glm::mat<DIMS, DIMS, float>(1) * 2.0f - 0.5f, grad);

            test_gaussian_amplitude_case(_cov<DIMS>(1.5f, -0.1f,  0.2f, 2.5f, -0.3f, 2.2f), grad);
            test_gaussian_amplitude_case(_cov<DIMS>(2.5f,  0.5f, -0.2f, 1.5f,  0.3f, 3.2f), grad);

        }
    }

    template<int DIMS>
    void test_gaussian_amplitude_case(const glm::mat<DIMS, DIMS, float>& cov, float grad) {
        auto cov_autodiff = gpe::makeAutodiff(cov);
        auto result_autodiff = gpe::gaussian_amplitude(cov_autodiff);
        result_autodiff.seed();
        result_autodiff.expr->propagate(grad);

        const auto grad_autodiff = gpe::extractGrad(cov_autodiff);
        const auto grad_analytical = gpe::grad::gaussian_amplitude(cov, grad);
        const glm::mat<DIMS, DIMS, float> diff = (grad_autodiff - grad_analytical);
        auto rmse = gpe::sqrt(gpe::sum(gpe::cwise_mul(diff, diff)));

        assert(rmse < 0.0001f);
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

    void test_outerProduct() {
        test_outerProduct_case<2>({1.f, 2.f}, {3.f, 4.f}, glm::mat<2, 2, float>(0, 0, 0, 0));
        test_outerProduct_case<2>({1.f, 2.f}, {3.f, 4.f}, glm::mat<2, 2, float>(1, 1, 1, 1));
        test_outerProduct_case<2>({1.f, 2.f}, {3.f, 4.f}, glm::mat<2, 2, float>(0.5, 1.5, 2.5, 3.5));
        test_outerProduct_case<2>({1.f, 2.f}, {3.f, 4.f}, glm::mat<2, 2, float>(-0.5, -1.5, -2.5, -3.5));

        test_outerProduct_case<3>({1.f, 2.f, 3.f}, {3.f, 4.f, 5.f}, glm::mat<3, 3, float>(0, 0, 0, 0, 0, 0, 0, 0, 0));
        test_outerProduct_case<3>({1.f, 2.f, 3.f}, {3.f, 4.f, 5.f}, glm::mat<3, 3, float>(1, 1, 1, 1, 1, 1, 1, 1, 1));
        test_outerProduct_case<3>({1.f, 2.f, 3.f}, {3.f, 4.f, 5.f}, glm::mat<3, 3, float>(0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5));
        test_outerProduct_case<3>({1.f, 2.f, 3.f}, {3.f, 4.f, 5.f}, glm::mat<3, 3, float>(-0.5, -1.5, -2.5, -3.5, -4.5, -5.5, -6.5, -7.5, -8.5));
        test_outerProduct_case<3>({1.f, 2.f, 3.f}, {3.f, 4.f, 5.f}, glm::mat<3, 3, float>(-0.5, -1.5, -2.5, -3.5, 4.5, 5.5, 6.5, 7.5, 8.5));
    }

    template<int DIMS>
    void test_outerProduct_case(const glm::vec<DIMS, float>& a,const glm::vec<DIMS, float>& b, const glm::mat<DIMS, DIMS, float>& grad) {
        test_binarycase(a, b, grad, outerProduct<DIMS>, gpe::grad::outerProduct<DIMS, float>);
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

    void test_scalarGrads_funs(float a, float b, float grad) {
//        test_scalarGrads_binarycase(a, b, grad, gpe::functors::times<AutodiffScalar>, gpe::grad::functors::times<float>);
        if (a > 0) {
            test_binarycase(a, b, grad, pow, gpe::grad::pow<float>);
        test_unarycase(a, grad, log, gpe::grad::log<float>);
        }

        test_unarycase(a, grad, exp, gpe::grad::exp<float>);

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
