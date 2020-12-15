#ifndef NDEBUG
#include <iostream>

#include "util/glm.h"

#include "util/autodiff.h"
#include "util/scalar.h"
#include "util/mixture.h"
#include "util/grad/mixture.h"
#include "util/grad/glm.h"


namespace  {
static struct UnitTests {
    UnitTests() {
        test_determinant<2>();
        test_determinant<3>();
        test_gaussian_amplitude<2>();
        test_gaussian_amplitude<3>();
        test_outerProduct();

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
        const auto grad_analytical = gpe::grad::gaussian_amplitude(cov, grad);
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
        const auto grad_analytical = gpe::grad::determinant(cov, grad);
        const glm::mat<DIMS, DIMS, float> diff = (grad_autodiff - grad_analytical);
        auto rmse = gpe::sqrt(gpe::sum(gpe::cwise_mul(diff, diff)) / (DIMS * DIMS));

        assert(rmse < 0.0001f);
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
        auto a_autodiff = gpe::makeAutodiff(a);
        auto b_autodiff = gpe::makeAutodiff(b);
        auto result_autodiff = glm::outerProduct(a_autodiff, b_autodiff);
        for (int i = 0; i < DIMS; ++i) {
            for (int j = 0; j < DIMS; ++j) {
                result_autodiff[i][j].expr->propagate(grad[i][j]);
            }
        }

        const auto grad_a_autodiff = gpe::extractGrad(a_autodiff);
        const auto grad_b_autodiff = gpe::extractGrad(b_autodiff);
        glm::vec<DIMS, float> grad_a{}, grad_b{};
        gpe::grad::outerProduct(a, b, &grad_a, &grad_b, grad);
        const glm::vec<DIMS, float> diff_a = (grad_a_autodiff - grad_a);
        const glm::vec<DIMS, float> diff_b = (grad_b_autodiff - grad_b);
        auto rmse_a = gpe::sqrt(gpe::sum(diff_a * diff_a) / (DIMS));
        auto rmse_b = gpe::sqrt(gpe::sum(diff_b * diff_b) / (DIMS));

        assert(rmse_a < 0.0001f);
        assert(rmse_b < 0.0001f);
    }

} unit_tests;

} // anonymous namespace

#endif // not NDEBUG
