#ifndef GPE_UTIL_GRAD_COMMON_H
#define GPE_UTIL_GRAD_COMMON_H

#include <cassert>

namespace gpe {
namespace grad {

template <typename A1, typename A2>
struct TwoGrads {
    A1 m_left;
    A2 m_right;

    void addTo(A1* left, A2* right) {
        cwise_ref_fun(&m_left, left, [](const auto& m_g, auto& g) { g += m_g; });
        cwise_ref_fun(&m_right, right, [](const auto& m_g, auto& g) { g += m_g; });
    }

    void addTo(A1* left, bool right) {
        assert(right == false);
        cwise_ref_fun(&m_left, left, [](const auto& m_g, auto& g) { g += m_g; });
    }

    void addTo(bool left, A2* right) {
        assert(left == false);
        cwise_ref_fun(&m_right, right, [](const auto& m_g, auto& g) { g += m_g; });
    }
};

template <typename A1>
struct OneGrad {
    A1 m_grad;
    void addTo(A1* grad) {
        cwise_ref_fun(&m_grad, grad, [](const auto& m_g, auto& g) { g += m_g; });
    }
};

} // namespace detail
} // namespace gpe
#endif // GPE_UTIL_GRAD_COMMON_H
