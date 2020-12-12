#ifndef NDEBUG
#include <iostream>

#include "math/gpe_glm.h"

#include "util/autodiff.h"
#include "math/scalar.h"
#include "containers.h"
#include "algorithms.h"
#include "util/algorithms_grad.h"


namespace  {

void assert_similar(float a, float b) {
    auto v = std::abs((a + b) / 2);
    v = (v > 0.00001f) ? v : 0.00001f;
    assert((std::abs(a - b) / v) < 0.00001f);
}

static struct UnitTests {
    UnitTests() {
        test_cwise();

        std::cout << "unit tests for algorithms_grad done" << std::endl;
    }

    void test_cwise() {
        std::vector<gpe::Array<float, 2>> shortArrs;
        shortArrs.push_back({0, 0});
        shortArrs.push_back({1, 1});
        shortArrs.push_back({1.2f, 0.5f});

        std::vector<gpe::Array<float, 4>> longArrs;
        longArrs.push_back({0, 0, 0, 0});
        longArrs.push_back({1, 1, 1, 1});
        longArrs.push_back({1.2f, 0.5f, -0.5f, -1.5f});


//        for (const auto& s1 : shortArrs) {
//            for (const auto& s2 : shortArrs) {
//                test_cwise_grads(s1, s2);
//            }
//        }
//        for (const auto& s1 : longArrs) {
//            for (const auto& s2 : longArrs) {
//                test_cwise_grads(s1, s2);
//            }
//        }

        for (const auto& s : shortArrs) {
            for (const auto& l : longArrs) {
                test_cwise_grads(s, gpe::outer_product(l, s, gpe::functors::times<float>));
                test_cwise_grads(gpe::outer_product(l, s, gpe::functors::times<float>), l);
                test_cwise_grads(gpe::outer_product(s, l, gpe::functors::times<float>), gpe::outer_product(s, l, gpe::functors::times<float>));
                test_cwise_grads(gpe::outer_product(l, s, gpe::functors::times<float>), gpe::outer_product(l, s, gpe::functors::times<float>));
            }
        }
    }

    template<typename A1, typename A2>
    void test_cwise_grads(const A1& left, const A2& right) {
        test_cwise_funs(left, right, gpe::cwise_fun(left, right, [](auto, auto) { return 0.f; }));
        test_cwise_funs(left, right, gpe::cwise_fun(left, right, [](auto, auto) { return 1.f; }));
        test_cwise_funs(left, right, gpe::cwise_fun(left, right, [](auto, auto) { return -1.f; }));
        test_cwise_funs(left, right, gpe::cwise_fun(left, right, [](auto, auto) { return 2.3f; }));
        test_cwise_funs(left, right, gpe::cwise_fun(left, right, [](auto, auto) { return -1.3f; }));
        float c = -1.0f;
        test_cwise_funs(left, right, gpe::cwise_fun(left, right, [&c](auto, auto) { c += 0.2f; return c; }));
        c = -100.f;
        test_cwise_funs(left, right, gpe::cwise_fun(left, right, [&c](auto, auto) { c += 0.2f; return c; }));
        c = -0.2f;
        test_cwise_funs(left, right, gpe::cwise_fun(left, right, [&c](auto, auto) { c += 0.2f; return c; }));
        c = 1.f;
        test_cwise_funs(left, right, gpe::cwise_fun(left, right, [&c](auto, auto) { c += 0.2f; return c; }));
    }

    template<typename A1, typename A2, typename A3>
    void test_cwise_funs(const A1& left, const A2& right, const A3& grad) {
        test_cwise_case(left, right, grad, gpe::functors::plus<autodiff::Variable<float>>, gpe::grad::functors::plus<float>);
        test_cwise_case(left, right, grad, gpe::functors::times<autodiff::Variable<float>>, gpe::grad::functors::times<float>);

        bool right_has_zeros = false;
        gpe::transform(right, [&right_has_zeros](float v) { if (gpe::abs(v) < 0.00001f) right_has_zeros = true; return 0; });
        if (!right_has_zeros)
            test_cwise_case(left, right, grad, gpe::functors::divided_AbyB<autodiff::Variable<float>>, gpe::grad::functors::divided_AbyB<float>);

        bool left_has_zeros = false;
        gpe::transform(left, [&left_has_zeros](float v) { if (gpe::abs(v) < 0.00001f) left_has_zeros = true; return 0; });
        if (!left_has_zeros)
            test_cwise_case(left, right, grad, gpe::functors::divided_BbyA<autodiff::Variable<float>>, gpe::grad::functors::divided_BbyA<float>);
    }

    template<typename A1, typename A2, typename A3, typename ForwardFun, typename GradFun>
    void test_cwise_case(const A1& left, const A2& right, const A3& grad, ForwardFun forward_fun, GradFun grad_fun) {
        auto left_autodiff = gpe::makeAutodiff(left);
        auto right_autodiff = gpe::makeAutodiff(right);
        auto result_autodiff = gpe::cwise_fun(left_autodiff, right_autodiff, forward_fun);
        gpe::cwise_fun(result_autodiff, grad, [](autodiff::Variable<float> r, float g) {
            r.expr->propagate(g);
            return 0;
        });

        auto grads = gpe::grad::cwise_fun(left, right, grad, grad_fun);

        gpe::cwise_fun(left_autodiff, grads.left_grad, [](const autodiff::Variable<float>& autodiff, const float& analytical) { assert_similar(autodiff.grad(), analytical); return 0; });
        gpe::cwise_fun(right_autodiff, grads.right_grad, [](const autodiff::Variable<float>& autodiff, const float& analytical) { assert_similar(autodiff.grad(), analytical); return 0; });
    }

} unit_tests;

} // anonymous namespace

#endif // not NDEBUG
