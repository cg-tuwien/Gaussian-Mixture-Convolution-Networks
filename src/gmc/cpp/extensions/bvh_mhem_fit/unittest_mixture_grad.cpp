#ifndef NDEBUG
#include <iostream>

#include "util/glm.h"

#include "util/autodiff.h"
#include "util/scalar.h"
#include "util/mixture.h"
#include "util/gaussian.h"
#include "util/grad/gaussian.h"
#include "util/grad/glm.h"


namespace  {
static struct UnitTests {
    UnitTests() {
        test_determinant<2>();
        test_determinant<3>();
        test_gaussian_amplitude<2>();
        test_gaussian_amplitude<3>();
        test_likelihood<2>();
        test_likelihood<3>();
        test_outerProduct();

        std::cout << "unit tests for mixture_grad done" << std::endl;
    }

    template<int DIMS>
    glm::vec<DIMS, float> _vec(float a, float b, float c) {
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
        auto t_autodiff = gpe::makeAutodiff(t);
        auto f_autodiff = gpe::makeAutodiff(f);
        auto result_autodiff = gpe::likelihood(t_autodiff, f_autodiff);
        result_autodiff.expr->propagate(grad);

        const auto grad_t_autodiff = gpe::extractGrad(t_autodiff);
        const auto grad_f_autodiff = gpe::extractGrad(f_autodiff);
        gpe::Gaussian<DIMS, float> grad_t_analytical;
        gpe::Gaussian<DIMS, float> grad_f_analytical;
        gpe::grad::likelihood(t, f, &grad_t_analytical, &grad_f_analytical, grad);


        auto ssrt = [](const auto& autodiff, const auto& analytical) {
            assert(std::abs(autodiff.weight - analytical.weight) < 0.00001f);
            for (int i = 0; i < DIMS; ++i) {
                assert(std::abs(autodiff.position[i] - analytical.position[i]) < 0.00001f);
            }
            const glm::mat<DIMS, DIMS, float> covdiff = (autodiff.covariance - analytical.covariance);
            auto rmse = gpe::sqrt(gpe::sum(gpe::cwise_mul(covdiff, covdiff)));
            assert(rmse < 0.0001f);
        };
        ssrt(grad_t_autodiff, grad_t_analytical);
        ssrt(grad_f_autodiff, grad_f_analytical);
    }

    template<int DIMS>
    void test_gaussian_amplitude() {
        for (float grad = -4.f; grad < 4.5f; grad++) {
            test_gaussian_amplitude_case(glm::mat<DIMS, DIMS, float>(1), grad);
            test_gaussian_amplitude_case(glm::mat<DIMS, DIMS, float>(1) * 0.5f, grad);
            test_gaussian_amplitude_case(glm::mat<DIMS, DIMS, float>(1) * 2.0f, grad);
            test_gaussian_amplitude_case(glm::mat<DIMS, DIMS, float>(1) * 2.0f + 0.5f, grad);
            test_gaussian_amplitude_case(glm::mat<DIMS, DIMS, float>(1) * 2.0f - 0.5f, grad);

            test_gaussian_amplitude_case(_cov<DIMS>(1.5f, -0.1f, 0.2f, 2.5f, -0.3f, 2.2f), grad);

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
