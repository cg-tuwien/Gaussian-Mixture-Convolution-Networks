#ifndef MATH_SCALAR_H
#define MATH_SCALAR_H

#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <algorithm>

#include <cuda_runtime.h>

#include "util/autodiff.h"

namespace gpe {

namespace detail {
// http://www.machinedlearnings.com/2011/06/fast-approximate-logarithm-exponential.html
// https://github.com/xodobox/fastapprox/blob/master/fastapprox/src/fastexp.h
// 2x faster, error in the range of e^-4 (dunno about relativ error)
static inline float fasterpow2 (float p)
{
    float clipp = (p < -126) ? -126.0f : p;
    union { uint32_t i; float f; } v = { uint32_t ( (1 << 23) * (clipp + 126.94269504f) ) };
    return v.f;
}

static inline float fasterexp (float p)
{
    return fasterpow2 (1.442695040f * p);
}

// slightly faster than std::exp, slightly less precise (error in the range of e-10)
static inline float
fastpow2 (float p)
{
    float offset = (p < 0) ? 1.0f : 0.0f;
    float clipp = (p < -126) ? -126.0f : p;
    int w = int(clipp);
    float z = clipp - float(w) + offset;
    union { uint32_t i; float f; } v = { uint32_t ( (1 << 23) * (clipp + 121.2740575f + 27.7280233f / (4.84252568f - z) - 1.49012907f * z) ) };

    return v.f;
}

static inline float
fastexp (float p)
{
    return fastpow2 (1.442695040f * p);
}

}

template <typename T> __forceinline__ __device__ __host__
T sq(T v) {
    return v * v;
}

#ifdef __CUDA_ARCH__
__forceinline__ __device__
float max(float a, float b) {
    return ::fmaxf(a, b);
}
__forceinline__ __device__
double max(double a, double b) {
    return ::fmax(a, b);
}
template <typename T> __forceinline__ __device__
T max(T a, T b) {
    return ::max(a, b);
}
__forceinline__ __device__
float min(float a, float b) {
    return ::fminf(a, b);
}
__forceinline__ __device__
double min(double a, double b) {
    return ::fmin(a, b);
}
template <typename T> __forceinline__ __device__
T min(T a, T b) {
    return ::min(a, b);
}
__forceinline__ __device__
float exp(float x) {
    return ::expf(x);
}
__forceinline__ __device__
double exp(double x) {
    return ::exp(x);
}
__forceinline__ __device__
float pow(float x, float y) {
    auto v = ::powf(x, y);
    // i'm leaving this assert in, as it can help finding surprising NaNs.
    // if fast math is in place, pow(0, 0) will give a NaN.
    // adding a small epsilon on x helps.
    assert(!::isnan(v));
    return v;
}
__forceinline__ __device__
double pow(double x, double y) {
    auto v = ::pow(x, y);
    assert(!::isnan(v));
    return v;
}

__forceinline__ __device__
float log(float x) {
    return ::logf(x);
}
__forceinline__ __device__
double log(double x) {
    return ::log(x);
}

__forceinline__ __device__
float sqrt(float x) {
    return ::sqrtf(x);
}
__forceinline__ __device__
double sqrt(double x) {
    return ::sqrt(x);
}

__forceinline__ __device__
float abs(float x) {
    return ::fabsf(x);
}
__forceinline__ __device__
double abs(double x) {
    return ::fabs(x);
}

__forceinline__ __device__
float acos(float x) {
    return ::acosf(x);
}
__forceinline__ __device__
double acos(double x) {
    return ::acos(x);
}

__forceinline__ __device__
float cos(float x) {
    return ::cosf(x);
}
__forceinline__ __device__
double cos(double x) {
    return ::cos(x);
}

template <typename T> __forceinline__ __device__
bool isnan(T x) {
    return ::isnan(x);
}

#else // __CUDA_ARCH__

template <typename scalar_t> inline
scalar_t max(scalar_t a, scalar_t b) {
    return std::max(a, b);
}

template <typename scalar_t> inline
scalar_t min(scalar_t a, scalar_t b) {
    return std::min(a, b);
}

template <typename scalar_t> inline
scalar_t exp(scalar_t x) {
    return std::exp(x);
}

template <typename scalar_t> inline
scalar_t pow(scalar_t x, scalar_t y) {
    return std::pow(x, y);
}

template <typename scalar_t> inline
scalar_t log(scalar_t x) {
    return std::log(x);
}

template <typename scalar_t> inline
scalar_t sqrt(scalar_t x) {
    return std::sqrt(x);
}

template <typename scalar_t> inline
scalar_t abs(scalar_t x) {
    return std::abs(x);
}

template <typename scalar_t> inline
scalar_t acos(scalar_t x) {
    return std::acos(x);
}

template <typename scalar_t> inline
scalar_t cos(scalar_t x) {
    return std::cos(x);
}


template <typename T>
inline bool isnan(T x) {
    return std::isnan(x);
}

#ifdef GPE_AUTODIFF
template <typename scalar_t>
using AutoDiffExpr = autodiff::reverse::ExprPtr<scalar_t>;
template <typename scalar_t>
using AutoDiffVariable = autodiff::Variable<scalar_t>;

template <typename scalar_t>
inline AutoDiffVariable<scalar_t> exp(AutoDiffExpr<scalar_t> x) {
    return autodiff::reverse::exp(x);
}

template <typename scalar_t>
inline AutoDiffVariable<scalar_t> pow(AutoDiffExpr<scalar_t> x, AutoDiffExpr<scalar_t> y) {
    return autodiff::reverse::pow(x, y);
}

template <typename scalar_t>
inline AutoDiffVariable<scalar_t> log(AutoDiffExpr<scalar_t> x) {
    return autodiff::reverse::log(x);
}

template <typename scalar_t>
inline AutoDiffVariable<scalar_t> sqrt(AutoDiffExpr<scalar_t> x) {
    return autodiff::reverse::sqrt(x);
}

template <typename scalar_t>
inline AutoDiffVariable<scalar_t> abs(AutoDiffExpr<scalar_t> x) {
    return autodiff::reverse::abs(x);
}

template <typename scalar_t>
inline bool isnan(AutoDiffExpr<scalar_t> x) {
    return std::isnan(x->val);
}

template <typename scalar_t>
inline AutoDiffVariable<scalar_t> exp(AutoDiffVariable<scalar_t> x) {
    return autodiff::reverse::exp(x);
}

template <typename scalar_t>
inline AutoDiffVariable<scalar_t> pow(AutoDiffVariable<scalar_t> x, AutoDiffVariable<scalar_t> y) {
    return autodiff::reverse::pow(x, y);
}
template <typename scalar_t>
inline AutoDiffVariable<scalar_t> pow(AutoDiffVariable<scalar_t> x, AutoDiffExpr<scalar_t> y) {
    return autodiff::reverse::pow(x.expr, y);
}
template <typename scalar_t>
inline AutoDiffVariable<scalar_t> pow(AutoDiffExpr<scalar_t> x, AutoDiffVariable<scalar_t> y) {
    return autodiff::reverse::pow(x, y.expr);
}

template <typename scalar_t>
inline AutoDiffVariable<scalar_t> log(AutoDiffVariable<scalar_t> x) {
    return autodiff::reverse::log(x);
}

template <typename scalar_t>
inline AutoDiffVariable<scalar_t> sqrt(AutoDiffVariable<scalar_t> x) {
    return autodiff::reverse::sqrt(x);
}

template <typename scalar_t>
inline AutoDiffVariable<scalar_t> abs(AutoDiffVariable<scalar_t> x) {
    return autodiff::reverse::abs(x);
}

template <typename scalar_t>
inline bool isnan(AutoDiffVariable<scalar_t> x) {
    return std::isnan(x.expr->val);
}
#endif //GPE_AUTODIFF
#endif //not __CUDA_ARCH__

template <typename scalar_t>
__host__ __device__ __forceinline__ int sign(scalar_t v) {
    return v >= 0 ? 1 : -1;
}

} // namespace gpe

#endif // HELPERS_H
