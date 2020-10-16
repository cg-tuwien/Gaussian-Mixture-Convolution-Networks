#ifndef MATH_SCALAR_H
#define MATH_SCALAR_H

#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <algorithm>

#include <cuda_runtime.h>

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

#ifdef __CUDA_ARCH__
__forceinline__ __device__ float max(float a, float b) {
    return ::fmaxf(a, b);
}
__forceinline__ __device__ float max(double a, double b) {
    return ::fmax(a, b);
}
template <typename T>
__forceinline__ __device__ T max(T a, T b) {
    return ::max(a, b);
}
__forceinline__ __device__ float min(float a, float b) {
    return ::fminf(a, b);
}
__forceinline__ __device__ float min(double a, double b) {
    return ::fmin(a, b);
}
template <typename T>
__forceinline__ __device__ T min(T a, T b) {
    return ::min(a, b);
}
__forceinline__ __device__ float exp(float x) {
    return ::expf(x);
}
__forceinline__ __device__ double exp(double x) {
    return ::exp(x);
}

__forceinline__ __device__ float log(float x) {
    return ::logf(x);
}
__forceinline__ __device__ double log(double x) {
    return ::log(x);
}

__forceinline__ __device__ float sqrt(float x) {
    return ::sqrtf(x);
}
__forceinline__ __device__ double sqrt(double x) {
    return ::sqrt(x);
}

__forceinline__ __device__ float abs(float x) {
    return ::fabsf(x);
}
__forceinline__ __device__ double abs(double x) {
    return ::fabs(x);
}
#else

template <typename scalar_t>
inline scalar_t max(scalar_t a, scalar_t b) {
    return std::max(a, b);
}

template <typename scalar_t>
inline scalar_t min(scalar_t a, scalar_t b) {
    return std::min(a, b);
}


template <typename scalar_t>
inline scalar_t exp(scalar_t x) {
    return std::exp(x);
}

template <typename scalar_t>
inline scalar_t log(scalar_t x) {
    return std::log(x);
}

template <typename scalar_t>
inline scalar_t sqrt(scalar_t x) {
    return std::sqrt(x);
}

template <typename scalar_t>
inline scalar_t abs(scalar_t x) {
    return std::abs(x);
}

#endif

}

#endif // HELPERS_H
