#ifndef CUDA_OPERATIONS_H
#define CUDA_OPERATIONS_H

#include <cuda_runtime.h>
#include <cstdint>


namespace gpe {
__host__ __device__ __forceinline__ int clz(const uint32_t value) noexcept
{
#ifdef __CUDA_ARCH__
    return ::__clz(value);
#elif defined (_MSC_VER)
    // untested:
    // from https://stackoverflow.com/questions/355967/how-to-use-msvc-intrinsics-to-get-the-equivalent-of-this-gcc-code
    unsigned long leading_zero = 0;
    if ( _BitScanReverse( &leading_zero, value ) )
    {
        return 31 - int(leading_zero);
    }
    else
    {
        // Same remarks as above
        return 32;
    }
#else
    return value ? __builtin_clz(value) : 32;
#endif
}
__host__ __device__
    inline int clz(const uint64_t value) noexcept
{
#ifdef __CUDA_ARCH__
    return ::__clzll(value);
#elif defined (_MSC_VER)
    // untested:
    // from https://stackoverflow.com/questions/355967/how-to-use-msvc-intrinsics-to-get-the-equivalent-of-this-gcc-code
    unsigned long leading_zero = 0;
    if ( _BitScanReverse64( &leading_zero, value ) )
    {
        return 31 - int(leading_zero);
    }
    else
    {
        // Same remarks as above
        return 64;
    }
#else
    return value ? __builtin_clzll(value) : 64;
#endif
}
}

#endif // CUDA_OPERATIONS_H
