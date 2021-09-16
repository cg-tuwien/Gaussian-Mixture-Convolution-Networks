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

#include "support.h"

using AutodiffScalar = autodiff::Variable<float>;

namespace {

struct UnitTests {
    template<int DIMS> static
    void test_likelihood() {
//        std::vector<float> grads = {-1.0, 0.f, 0.4f, 1.f};
//        for (auto grad : grads) {
//            for (auto g1 : _gaussianCollection<DIMS>()) {
//                for (auto g2 : _gaussianCollection<DIMS>()) {
//                    test_binarycase(g1, g2, grad, gpe::likelihood<AutodiffScalar, DIMS>, gpe::grad::likelihood<float, DIMS>);
//                }
//            }
//        }
    }

    template<int DIMS> static
    void test_gaussian_to_scalar_functions() {
        for (auto grad : _scalarCollection()) {
            for (auto g : _gaussianCollection<DIMS>()) {
                for (auto p : _vecCollection<DIMS>()) {
                    test_binarycase(g, p, grad, gpe::evaluate<AutodiffScalar, DIMS>, gpe::grad::evaluate<float, DIMS>);
                }
            }
        }
    }

    template<int DIMS> static
    void test_gaussian_to_gaussian_functions() {
//        for (auto grad : _gaussianCollection2<DIMS>()) {
//            for (auto g : _gaussianCollection<DIMS>()) {
//                for (auto p : _gaussianCollection<DIMS>()) {
//                    test_binarycase(g, p, grad, gpe::convolve<AutodiffScalar, DIMS>, gpe::grad::convolve<float, DIMS>);
//                }
//            }
//        }
    }

    template<int DIMS> static
    void test_cov_functions() {
        for (float grad : _scalarCollection()) {
            for (const auto& cov : _covCollection<DIMS>()) {
                test_unarycase(cov, grad, determinant<DIMS>, gpe::grad::determinant<float, DIMS>);
            }
        }
    }

    template<int DIMS> static
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

    static
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

    template<int N_DIMS> static
    glm::mat<N_DIMS, N_DIMS, AutodiffScalar> outerProduct(const glm::vec<N_DIMS, AutodiffScalar>& a, const glm::vec<N_DIMS, AutodiffScalar>& b) { return glm::outerProduct(a, b); }

    template<int N_DIMS> static
    AutodiffScalar determinant(const glm::mat<N_DIMS, N_DIMS, AutodiffScalar>& m) { return glm::determinant(m); }

    template<int N_DIMS> static
    glm::mat<N_DIMS, N_DIMS, AutodiffScalar> matrix_inverse(const glm::mat<N_DIMS, N_DIMS, AutodiffScalar>& m) { return glm::inverse(m); }

    template<int N_DIMS> static
    AutodiffScalar dot(const glm::vec<N_DIMS, AutodiffScalar>& a, const glm::vec<N_DIMS, AutodiffScalar>& b) { return glm::dot(a, b); }

    static
    void test_scalarGrads_funs(float a, float b, float grad) {
//        test_scalarGrads_binarycase(a, b, grad, gpe::functors::times<AutodiffScalar>, gpe::grad::functors::times<float>);
        if (a > 0) {
            test_binarycase(a, b, grad, pow, gpe::grad::pow<float>);
        test_unarycase(a, grad, log, gpe::grad::log<float>);
        }

        test_unarycase(a, grad, exp, gpe::grad::exp<float>);
    }

    template<int N_DIMS> static
    void test_matrix_inverse() {
        for (auto grad : _covCollection<N_DIMS>()) {
            for (auto cov : _covCollection<N_DIMS>()) {
                test_unarycase(cov, grad, matrix_inverse<N_DIMS>, gpe::grad::inverse<float, N_DIMS>);
            }
        }
    }
};

}


TEST_CASE("grad mixture") {
    SECTION("cov functions") {
        UnitTests::test_cov_functions<2>();
        UnitTests::test_cov_functions<3>();
    }
    SECTION("gaussian to scalar functions") {
        UnitTests::test_gaussian_to_scalar_functions<2>();
        UnitTests::test_gaussian_to_scalar_functions<3>();
    }
    SECTION("gaussian to gaussian functions") {
        UnitTests::test_gaussian_to_gaussian_functions<2>();
        UnitTests::test_gaussian_to_gaussian_functions<3>();
    }
    SECTION("likelihood") {
        UnitTests::test_likelihood<2>();
        UnitTests::test_likelihood<3>();
    }
    SECTION("vec on vec functions") {
        UnitTests::test_vecOnVec<2>();
        UnitTests::test_vecOnVec<3>();
    }
    SECTION("scalar functions") {
        UnitTests::test_scalarGrads();
    }
    SECTION("matrix functions") {
        UnitTests::test_matrix_inverse<2>();
        UnitTests::test_matrix_inverse<3>();
    }
}
