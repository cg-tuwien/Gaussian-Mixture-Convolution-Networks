#ifndef MATH_SCALAR_H
#define MATH_SCALAR_H

#include <cstdint>
#include <cmath>

//#include <cuda.h>
//#include <cuda_runtime.h>


#ifndef __CUDACC__
#define __device__
#define __host__
#endif

#ifndef __forceinline__
#define __forceinline__ inline
#endif

using uint = unsigned int;

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


__forceinline__ __device__ float exp(float x) {
    return ::expf(x);
}
__forceinline__ __device__ double exp(double x) {
    return ::exp(x);
}

__forceinline__ __device__ float sqrt(float x) {
    return ::sqrtf(x);
}
__forceinline__ __device__ double sqrt(double x) {
    return ::sqrt(x);
}

template <typename scalar_t>
inline scalar_t exp(scalar_t x) {
    return std::exp(x);
}

template <typename scalar_t>
inline scalar_t sqrt(scalar_t x) {
    return std::sqrt(x);
}


}

#endif // HELPERS_H
