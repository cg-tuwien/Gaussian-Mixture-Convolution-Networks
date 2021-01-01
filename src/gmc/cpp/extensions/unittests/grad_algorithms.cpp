#include <catch2/catch.hpp>

#include "util/glm.h"
#include "util/autodiff.h"
#include "util/scalar.h"
#include "util/containers.h"
#include "util/algorithms.h"
#include "util/grad/algorithms.h"

#include "support.h"

template<uint32_t N1, uint32_t N2, typename ForwardFun, typename GradFun>
void test_outer_product_case(const gpe::Array<float, N1>& left, const gpe::Array<float, N2>& right, const gpe::Array2d<float, N1, N2>& grad, ForwardFun forward_fun, GradFun grad_fun) {
    auto left_autodiff = gpe::makeAutodiff(left);
    auto right_autodiff = gpe::makeAutodiff(right);
    auto result_autodiff = gpe::outer_product(left_autodiff, right_autodiff, forward_fun);
    gpe::cwise_fun(result_autodiff, grad, [](autodiff::Variable<float> r, float g) {
        r.expr->propagate(g);
        return 0;
    });

    auto grads = gpe::grad::outer_product(left, right, grad, grad_fun);

    gpe::cwise_fun(left_autodiff, grads.m_left, [](const autodiff::Variable<float>& autodiff, const float& analytical) { REQUIRE(are_similar(autodiff.grad(), analytical)); return 0; });
    gpe::cwise_fun(right_autodiff, grads.m_right, [](const autodiff::Variable<float>& autodiff, const float& analytical) { REQUIRE(are_similar(autodiff.grad(), analytical)); return 0; });
}

template<typename A1, typename A2, typename A3>
void test_outer_product_funs(const A1& left, const A2& right, const A3& grad) {
    test_outer_product_case(left, right, grad, gpe::functors::plus<autodiff::Variable<float>>, gpe::grad::functors::plus<float>);
    test_outer_product_case(left, right, grad, gpe::functors::minus<autodiff::Variable<float>>, gpe::grad::functors::minus<float>);
    test_outer_product_case(left, right, grad, gpe::functors::times<autodiff::Variable<float>>, gpe::grad::functors::times<float>);

    bool right_has_zeros = false;
    gpe::transform(right, [&right_has_zeros](float v) { if (gpe::abs(v) < 0.00001f) right_has_zeros = true; return 0; });
    if (!right_has_zeros)
        test_outer_product_case(left, right, grad, gpe::functors::divided_AbyB<autodiff::Variable<float>>, gpe::grad::functors::divided_AbyB<float>);

    bool left_has_zeros = false;
    gpe::transform(left, [&left_has_zeros](float v) { if (gpe::abs(v) < 0.00001f) left_has_zeros = true; return 0; });
    if (!left_has_zeros)
        test_outer_product_case(left, right, grad, gpe::functors::divided_BbyA<autodiff::Variable<float>>, gpe::grad::functors::divided_BbyA<float>);
}

template<typename A1, typename A2>
void test_outer_product_grads(const A1& left, const A2& right) {
    test_outer_product_funs(left, right, gpe::outer_product(left, right, [](auto, auto) { return 0.f; }));
    test_outer_product_funs(left, right, gpe::outer_product(left, right, [](auto, auto) { return 1.f; }));
    test_outer_product_funs(left, right, gpe::outer_product(left, right, [](auto, auto) { return -1.f; }));
    test_outer_product_funs(left, right, gpe::outer_product(left, right, [](auto, auto) { return 2.3f; }));
    test_outer_product_funs(left, right, gpe::outer_product(left, right, [](auto, auto) { return -1.3f; }));
    float c = -2.0f;
    test_outer_product_funs(left, right, gpe::outer_product(left, right, [&c](auto, auto) { c += 0.2f; return c; }));
    c = -100.f;
    test_outer_product_funs(left, right, gpe::outer_product(left, right, [&c](auto, auto) { c += 0.2f; return c; }));
    c = -0.55f;
    test_outer_product_funs(left, right, gpe::outer_product(left, right, [&c](auto, auto) { c += 0.2f; return c; }));
    c = 1.f;
    test_outer_product_funs(left, right, gpe::outer_product(left, right, [&c](auto, auto) { c += 0.2f; return c; }));
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

    gpe::cwise_fun(left_autodiff, grads.m_left, [](const autodiff::Variable<float>& autodiff, const float& analytical) { REQUIRE(are_similar(autodiff.grad(), analytical)); return 0; });
    gpe::cwise_fun(right_autodiff, grads.m_right, [](const autodiff::Variable<float>& autodiff, const float& analytical) { REQUIRE(are_similar(autodiff.grad(), analytical)); return 0; });
}

template<typename A1, typename A2, typename A3>
void test_cwise_funs(const A1& left, const A2& right, const A3& grad) {
    test_cwise_case(left, right, grad, gpe::functors::plus<autodiff::Variable<float>>, gpe::grad::functors::plus<float>);
    test_cwise_case(left, right, grad, gpe::functors::minus<autodiff::Variable<float>>, gpe::grad::functors::minus<float>);
    test_cwise_case(left, right, grad, gpe::functors::times<autodiff::Variable<float>>, gpe::grad::functors::times<float>);

    bool right_contains_zeros = false;
    gpe::transform(right, [&right_contains_zeros](float v) { if (gpe::abs(v) < 0.00001f) right_contains_zeros = true; return 0; });
    if (!right_contains_zeros)
        test_cwise_case(left, right, grad, gpe::functors::divided_AbyB<autodiff::Variable<float>>, gpe::grad::functors::divided_AbyB<float>);

    bool left_contains_zeros = false;
    gpe::transform(left, [&left_contains_zeros](float v) { if (gpe::abs(v) < 0.00001f) left_contains_zeros = true; return 0; });
    if (!left_contains_zeros)
        test_cwise_case(left, right, grad, gpe::functors::divided_BbyA<autodiff::Variable<float>>, gpe::grad::functors::divided_BbyA<float>);
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


TEST_CASE("grad algorithms") {
    SECTION( "outer product", "[grad_algorithms]" ) {
        std::vector<gpe::Array<float, 2>> shortArrs;
        shortArrs.push_back({0, 0});
        shortArrs.push_back({1, 1});
        shortArrs.push_back({1.2f, 0.5f});

        std::vector<gpe::Array<float, 4>> longArrs;
        longArrs.push_back({0, 0, 0, 0});
        longArrs.push_back({1, 1, 1, 1});
        longArrs.push_back({1.2f, 0.5f, -0.5f, -1.5f});


        for (const auto& s1 : shortArrs) {
            for (const auto& s2 : shortArrs) {
                test_outer_product_grads(s1, s2);
            }
        }
        for (const auto& s1 : longArrs) {
            for (const auto& s2 : longArrs) {
                test_outer_product_grads(s1, s2);
            }
        }
        for (const auto& s1 : shortArrs) {
            for (const auto& s2 : longArrs) {
                test_outer_product_grads(s1, s2);
            }
        }
        for (const auto& s1 : longArrs) {
            for (const auto& s2 : shortArrs) {
                test_outer_product_grads(s1, s2);
            }
        }
    }

    SECTION( "cwise ", "[grad_algorithms]" ) {
        std::vector<gpe::Array<float, 2>> shortArrs;
        shortArrs.push_back({0, 0});
        shortArrs.push_back({1, 1});
        shortArrs.push_back({1.2f, 0.5f});

        std::vector<gpe::Array<float, 4>> longArrs;
        longArrs.push_back({0, 0, 0, 0});
        longArrs.push_back({1, 1, 1, 1});
        longArrs.push_back({1.2f, 0.5f, -0.5f, -1.5f});


        for (const auto& s1 : shortArrs) {
            for (const auto& s2 : shortArrs) {
                test_cwise_grads(s1, s2);
            }
        }
        for (const auto& s1 : longArrs) {
            for (const auto& s2 : longArrs) {
                test_cwise_grads(s1, s2);
            }
        }

        for (const auto& s : shortArrs) {
            for (const auto& l : longArrs) {
                test_cwise_grads(s, gpe::outer_product(l, s, gpe::functors::times<float>));
                test_cwise_grads(gpe::outer_product(l, s, gpe::functors::times<float>), l);
                test_cwise_grads(gpe::outer_product(s, l, gpe::functors::times<float>), gpe::outer_product(s, l, gpe::functors::times<float>));
                test_cwise_grads(gpe::outer_product(l, s, gpe::functors::times<float>), gpe::outer_product(l, s, gpe::functors::times<float>));
            }
        }
    }

    SECTION( "manual differentiation approach ", "[grad_algorithms]" ) {
        namespace fun = gpe::functors;
        namespace gradfun = gpe::grad::functors;

        gpe::Array<float, 8> a = {0, 1, 2, 3, 4, 5, 6, 7};
        gpe::Array<float, 4> b = {0.1f, 0.2f, 0.3f, 0.4f};

        auto ab = gpe::outer_product(a, b, fun::times<float>);
        gpe::Array<float, 4> c = {-1.1f, -1.2f, -1.3f, -1.4f};

        const auto cc = gpe::cwise_fun(c, c, fun::times<float>);
        const auto ab_plus_cc = gpe::cwise_fun(cc, ab, fun::plus<float>);
        const auto ab_plus_cc_times_cc = gpe::cwise_fun(cc, ab_plus_cc, fun::times<float>);
        const auto ab_plus_cc_times_cc_sq = gpe::cwise_fun(ab_plus_cc_times_cc, ab_plus_cc_times_cc, fun::times<float>);
        const auto sum = gpe::reduce(ab_plus_cc_times_cc_sq, 0.f, fun::plus<float>);

        // define grads
        std::decay_t<decltype (sum)> sum_grad = 1.5f;
        std::decay_t<decltype (ab_plus_cc_times_cc_sq)> ab_plus_cc_times_cc_sq_grad{};
        std::decay_t<decltype (ab_plus_cc_times_cc)> ab_plus_cc_times_cc_grad{};
        std::decay_t<decltype (ab_plus_cc)> ab_plus_cc_grad{};
        std::decay_t<decltype (cc)> cc_grad{};
        std::decay_t<decltype (c)> c_grad{};
        std::decay_t<decltype (ab)> ab_grad{};

        // compute analytical grads
        gpe::grad::sum(ab_plus_cc_times_cc_sq, sum_grad).addTo(&ab_plus_cc_times_cc_sq_grad);
        gpe::grad::cwise_fun(ab_plus_cc_times_cc, ab_plus_cc_times_cc, ab_plus_cc_times_cc_sq_grad, gradfun::times<float>).addTo(&ab_plus_cc_times_cc_grad, &ab_plus_cc_times_cc_grad);
        gpe::grad::cwise_fun(cc, ab_plus_cc, ab_plus_cc_times_cc_grad, gradfun::times<float>).addTo(&cc_grad, &ab_plus_cc_grad);
        gpe::grad::cwise_fun(cc, ab, ab_plus_cc_grad, gradfun::plus<float>).addTo(&cc_grad, &ab_grad);
        gpe::grad::cwise_fun(c, c, cc_grad, gradfun::times<float>).addTo(&c_grad, &c_grad);

        // compute autodiff grads
        {
            auto ab_autodiff = gpe::makeAutodiff(ab);
            auto c_autodiff = gpe::makeAutodiff(c);
            using AdFloat = autodiff::Variable<float>;
            auto cc = gpe::cwise_fun(c_autodiff, c_autodiff, fun::times<AdFloat, AdFloat, AdFloat>);
            auto ab_plus_cc = gpe::cwise_fun(cc, ab_autodiff, fun::plus<AdFloat, AdFloat, AdFloat>);
            auto ab_plus_cc_times_cc = gpe::cwise_fun(cc, ab_plus_cc, fun::times<AdFloat, AdFloat, AdFloat>);
            auto ab_plus_cc_times_cc_sq = gpe::cwise_fun(ab_plus_cc_times_cc, ab_plus_cc_times_cc, fun::times<AdFloat, AdFloat, AdFloat>);
            auto sum = gpe::reduce(ab_plus_cc_times_cc_sq, AdFloat(0.f), fun::plus<AdFloat, AdFloat, AdFloat>);
            sum.expr->propagate(sum_grad);

            // compare
            gpe::cwise_fun(ab_autodiff, ab_grad, [](const autodiff::Variable<float>& autodiff, const float& analytical) { REQUIRE(are_similar(autodiff.grad(), analytical)); return 0; });
            gpe::cwise_fun(c_autodiff, c_grad, [](const autodiff::Variable<float>& autodiff, const float& analytical) { REQUIRE(are_similar(autodiff.grad(), analytical)); return 0; });
        }
    }
}

