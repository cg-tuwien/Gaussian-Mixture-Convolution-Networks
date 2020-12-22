#ifndef NDEBUG
#include <iostream>

#include "unit_test_support.h"
#include "util/algorithms.h"
#include "util/autodiff.h"
#include "util/mixture.h"
#include "util/gaussian.h"
#include "util/glm.h"
#include "util/grad/algorithms.h"
#include "util/grad/gaussian.h"
#include "util/grad/scalar.h"
#include "util/grad/glm.h"
#include "util/scalar.h"


namespace  {
using AutodiffScalar = autodiff::Variable<float>;

static struct UnitTests {
    UnitTests() {
        test_cov_functions<2>();
        test_cov_functions<3>();
        test_gaussian_functions<2>();
        test_gaussian_functions<3>();
        test_likelihood<2>();
        test_likelihood<3>();
        test_vecOnVec<2>();
        test_vecOnVec<3>();
        test_scalarGrads();
        test_matrix_inverse<2>();
        test_matrix_inverse<3>();

        std::cout << "unit tests for mixture_grad done" << std::endl;
    }


    template<int DIMS>
    void test_likelihood() {
        std::vector<float> grads = {-1.0, 0.f, 0.4f, 1.f};
        for (auto grad : grads) {
            for (auto g1 : _gaussianCollection<DIMS>()) {
                for (auto g2 : _gaussianCollection<DIMS>()) {
                    test_binarycase(g1, g2, grad, gpe::likelihood<AutodiffScalar, DIMS>, gpe::grad::likelihood<float, DIMS>);
                }
            }
        }
    }

    template<int DIMS>
    void test_gaussian_functions() {
        for (auto grad : _scalarCollection()) {
            for (auto g : _gaussianCollection<DIMS>()) {
                test_unarycase(g, grad, gpe::integrate<AutodiffScalar, DIMS>, gpe::grad::integrate<float, DIMS>);
                for (auto p : _vecCollection<DIMS>()) {
                    test_binarycase(g, p, grad, gpe::evaluate<AutodiffScalar, DIMS>, gpe::grad::evaluate<float, DIMS>);
                }
            }
        }
    }

    template<int DIMS>
    void test_cov_functions() {
        for (float grad : _scalarCollection()) {
            for (const auto& cov : _covCollection<DIMS>()) {
                test_unarycase(cov, grad, gpe::gaussian_amplitude<AutodiffScalar, DIMS>, gpe::grad::gaussian_amplitude<float, DIMS>);
                test_unarycase(cov, grad, determinant<DIMS>, gpe::grad::determinant<float, DIMS>);
            }
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


} unit_tests;

} // anonymous namespace

#endif // not NDEBUG
