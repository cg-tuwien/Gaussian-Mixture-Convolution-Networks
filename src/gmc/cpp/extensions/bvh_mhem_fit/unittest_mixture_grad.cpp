#ifndef NDEBUG
#include <iostream>

#include "math/gpe_glm.h"

#include "util/autodiff.h"
#include "math/scalar.h"
#include "mixture.h"


namespace  {
static struct UnitTests {
    UnitTests() {
        test_determinant<2>();
        test_determinant<3>();
        test_gaussian_amplitude<2>();
        test_gaussian_amplitude<3>();

        std::cout << "unit tests for mixture_grad done" << std::endl;
    }

    template<int DIMS>
    void test_gaussian_amplitude() {
        for (float grad = -4.f; grad < 4.5f; grad++) {
            test_gaussian_amplitude_case(glm::mat<DIMS, DIMS, float>(1), grad);
            test_gaussian_amplitude_case(glm::mat<DIMS, DIMS, float>(1) * 0.5f, grad);
            test_gaussian_amplitude_case(glm::mat<DIMS, DIMS, float>(1) * 2.0f, grad);
            test_gaussian_amplitude_case(glm::mat<DIMS, DIMS, float>(1) * 2.0f + 0.5f, grad);
            test_gaussian_amplitude_case(glm::mat<DIMS, DIMS, float>(1) * 2.0f - 0.5f, grad);

            test_gaussian_amplitude_case(glm::mat<DIMS, DIMS, float>({{1.5, -0.1}, {-0.1, 2.5}}), grad);
        }
    }

    template<int DIMS>
    void test_gaussian_amplitude_case(const glm::mat<DIMS, DIMS, float>& cov, float grad) {
        auto cov_autodiff = gpe::makeAutodiff(cov);
        auto result_autodiff = gpe::gaussian_amplitude(cov_autodiff);
        result_autodiff.seed();
        result_autodiff.expr->propagate(grad);

        const auto grad_autodiff = gpe::extractGrad(cov_autodiff);
        const auto grad_analytical = gpe::grad_gaussian_amplitude(cov, grad);
        const glm::mat<DIMS, DIMS, float> diff = (grad_autodiff - grad_analytical);
        auto rmse = gpe::sqrt(gpe::sum(gpe::cwise_mul(diff, diff)));

        assert(rmse < 0.0001f);
    }

    template<int DIMS>
    void test_determinant() {
        for (float grad = -4.f; grad < 4.5f; grad++) {
            test_determinant_case(glm::mat<DIMS, DIMS, float>(1), grad);
            test_determinant_case(glm::mat<DIMS, DIMS, float>(1) * 0.5f, grad);
            test_determinant_case(glm::mat<DIMS, DIMS, float>(1) * 2.0f, grad);
            test_determinant_case(glm::mat<DIMS, DIMS, float>(1) * 2.0f + 0.5f, grad);
            test_determinant_case(glm::mat<DIMS, DIMS, float>(1) * 2.0f - 0.5f, grad);

            test_determinant_case(glm::mat<DIMS, DIMS, float>({{1.5, -0.1}, {-0.1, 2.5}}), grad);
        }
    }

    template<int DIMS>
    void test_determinant_case(const glm::mat<DIMS, DIMS, float>& cov, float grad) {
        auto cov_autodiff = gpe::makeAutodiff(cov);
        auto result_autodiff = glm::determinant(cov_autodiff);
        result_autodiff.seed();
        result_autodiff.expr->propagate(grad);

        const auto grad_autodiff = gpe::extractGrad(cov_autodiff);
        const auto grad_analytical = gpe::grad_determinant(cov, grad);
        const glm::mat<DIMS, DIMS, float> diff = (grad_autodiff - grad_analytical);
        auto rmse = gpe::sqrt(gpe::sum(gpe::cwise_mul(diff, diff)) / (DIMS * DIMS));

        assert(rmse < 0.0001f);
    }

} unit_tests;

} // anonymous namespace

#endif // not NDEBUG
