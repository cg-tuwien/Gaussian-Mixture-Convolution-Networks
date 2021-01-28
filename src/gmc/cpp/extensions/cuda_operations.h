#ifndef CUDA_OPERATIONS_H
#define CUDA_OPERATIONS_H

#include <atomic>
#include <bitset>
#include <cstdint>

#include <cuda_runtime.h>

#include "CpuSynchronisationPoint.h"

#ifdef __CUDA_ARCH__
#define GPE_SHARED __shared__
#else
#define GPE_SHARED static
#endif

namespace gpe {
namespace detail {
inline bool is_aligned(std::size_t alignment, const volatile void* ptr)
{
    return (reinterpret_cast<std::size_t>(ptr) & (alignment - 1)) == 0;
}
}
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

__host__ __device__ __forceinline__ uint32_t popc(uint32_t v) {
#ifdef __CUDA_ARCH__
    return __popc(v);
#else
    return std::bitset<32>(v).count();
#endif
}

/// sync_id is used only on cpu side, required to deal with spurious wakeups. every syncthreads needs a unique sync_id.
__host__ __device__ __forceinline__ void syncthreads(unsigned sync_id) {
#ifdef __CUDA_ARCH__
    __syncthreads();
#else
    gpe::detail::CpuSynchronisationPoint::synchronise(sync_id);
#endif
}

/// sync_id is used only on cpu side, required to deal with spurious wakeups. every syncthreads needs a unique sync_id.
__host__ __device__ __forceinline__ void syncwarp(unsigned sync_id) {
#ifdef __CUDA_ARCH__
    __syncwarp();
#else
    gpe::detail::CpuSynchronisationPoint::synchronise(sync_id);
#endif
}

template<typename Assignable1, typename Assignable2>
__host__ __device__ __forceinline__
void swap(Assignable1 &a, Assignable2 &b)
{
    Assignable1 temp = a;
    a = b;
    b = temp;
}


template <class T>
__host__ __device__ __forceinline__ void atomicAdd(T *ptr, T val) {
//    *ptr += val;
//    return *ptr;
#ifdef __CUDA_ARCH__
    ::atomicAdd(ptr, val);
#elif defined(_OPENMP)
#pragma omp atomic
    *ptr += val;
#else
#error "Requires OpenMP"
#endif
}

/// sync_id is used only on cpu side, required to deal with spurious wakeups. every ballot_sync needs a unique sync_id (and mustn't be shared with other calls to syncthreads).
__host__ __device__ __forceinline__ uint32_t ballot_sync(uint32_t mask, bool predicate, uint32_t thread_id, unsigned sync_id) {
#ifdef __CUDA_ARCH__
    return __ballot_sync(mask, predicate);
#else
    assert(thread_id < 32);
    assert(sync_id < 100000);
    static std::atomic_uint32_t ballot;
    ballot.store(0);

    syncthreads(sync_id);

//    printf("thread_id=%d; sync_id=%d; ballot.load() = %x\n", thread_id, sync_id, ballot.load());
    uint32_t thread_mask = (uint32_t(predicate) << thread_id);
    uint32_t or_mask = mask & thread_mask;

//    syncthreads(sync_id + 100000);

//    printf("thread_id=%d; sync_id=%d; or_mask = %x\n", thread_id, sync_id, or_mask);
    ballot.fetch_or(or_mask);

    syncthreads(sync_id + 200000);

    auto result = ballot.load();
//    printf("thread_id=%d; sync_id=%d; result = %x\n", thread_id, sync_id, result);

    // this sync must stay, otherwise there can be a race via ballot.store(0);
    syncthreads(sync_id + 300000);

    return result;
#endif
}

template <class T>
__host__ __device__ __forceinline__ T atomicCAS(T *addr, T compare, T val) {

#ifdef __CUDA_ARCH__
    return ::atomicCAS(addr, compare, val);
#else
    // undefined, but works on gcc 10.2, 9.3, clang 10, 11, and msvc 19.27
    // https://godbolt.org/z/fGK77j
    assert(detail::is_aligned(std::max(sizeof(T), (size_t)4ul), addr));   // afaik *addr must be aligned
    auto d = reinterpret_cast<std::atomic_int32_t*>(addr);
    d->compare_exchange_strong(compare, val);
    return compare;
#endif
}
}

#endif // CUDA_OPERATIONS_H
