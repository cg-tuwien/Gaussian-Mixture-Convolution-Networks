#ifndef PARALLEL_START_H
#define PARALLEL_START_H

#include <atomic>
#include <cassert>
#include <iostream>
#include <string>
#include <omp.h>

#include <cuda_runtime.h>
#include <torch/types.h>

#include "cuda_qt_creator_definitinos.h"


// replacement for AT_DISPATCH_FLOATING_TYPES
#define GPE_PRIVATE_CASE_TYPE_AND_DIM(enum_type, type, n_dims, ...) \
  case enum_type: {                                                 \
    using scalar_t = type;                                          \
    if (n_dims == 2) {                                              \
        constexpr int N_DIMS = 2;                                   \
        return __VA_ARGS__();                                       \
    }                                                               \
    else if (n_dims == 3) {                                         \
        constexpr int N_DIMS = 3;                                   \
        return __VA_ARGS__();                                       \
    }                                                               \
    else {                                                          \
        std::string dimstr = std::to_string(n_dims);                \
        AT_ERROR(__FILE__, ":", __LINE__, " not implemented for 'n_dims == ", dimstr.c_str(), "'"); \
    }                                                               \
  }

#define GPE_DISPATCH_FLOATING_TYPES_AND_DIM(TYPE, N_DIMS, ...)                          \
  [&] {                                                                                 \
    const auto& the_type = TYPE;                                                        \
    const auto& the_n_dims = N_DIMS;                                                    \
    at::ScalarType _st = ::detail::scalar_type(the_type);                               \
    switch (_st) {                                                                      \
      GPE_PRIVATE_CASE_TYPE_AND_DIM(at::ScalarType::Double, double, the_n_dims, __VA_ARGS__)    \
      GPE_PRIVATE_CASE_TYPE_AND_DIM(at::ScalarType::Float, float, the_n_dims, __VA_ARGS__)      \
      default:                                                                          \
        AT_ERROR(__FILE__, ":", __LINE__, " not implemented for '", at::toString(_st), "'"); \
    }                                                                                   \
  }()

namespace gpe {

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


template <class T>
__host__ __device__ __forceinline__ T atomicCAS(T *addr, T compare, T val) {

#ifdef __CUDA_ARCH__
    return ::atomicCAS(addr, compare, val);
#else
    // undefined, but works on gcc 10.2, 9.3, clang 10, 11, and msvc 19.27
    // https://godbolt.org/z/fGK77j
    auto d = reinterpret_cast<std::atomic_int32_t*>(addr);
    d->compare_exchange_strong(compare, val);
    return compare;
#endif
}

namespace detail {

//void gpe_start_cpu_parallel(const dim3& gridDim, const dim3& blockDim, const std::function<void(const dim3&, const dim3&, const dim3&, const dim3&)>& function);
template <typename Fun>
inline void gpe_start_cpu_parallel(const dim3& gridDim, const dim3& blockDim, Fun function) {
    #pragma omp parallel for num_threads(omp_get_num_procs()) collapse(3)
    for (int blockIdxZ = 0; blockIdxZ < gridDim.z; ++blockIdxZ) {
        for (int blockIdxY = 0; blockIdxY < gridDim.y; ++blockIdxY) {
            for (int blockIdxX = 0; blockIdxX < gridDim.x; ++blockIdxX) {
                const auto blockIdx = dim3{unsigned(blockIdxX), unsigned(blockIdxY), unsigned(blockIdxZ)};
                dim3 threadIdx = {0, 0, 0};
                for (threadIdx.z = 0; threadIdx.z < blockDim.z; ++threadIdx.z) {
                    for (threadIdx.y = 0; threadIdx.y < blockDim.y; ++threadIdx.y) {
                        for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) {
                            function(gridDim, blockDim, blockIdx, threadIdx);
                        }
                    }
                }
            }
        }
    }

}

#ifdef __CUDACC__
template <typename Fun>
__global__ void gpe_generic_cuda_kernel(Fun function) {
    function(gridDim, blockDim, blockIdx, threadIdx);
}

inline void gpu_assert(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert in start_parallel: %s\n", cudaGetErrorString(code));
        exit(code);
    }
}
#endif
} // namespace detail

enum class ComputeDevice {
    CPU, CUDA, Both
};

inline ComputeDevice device(const torch::Tensor& t) {
    return t.is_cuda() ? ComputeDevice::CUDA : ComputeDevice::CPU;
}

template <ComputeDevice allowed_devices, typename Fun>
void start_parallel(ComputeDevice device, const dim3& gridDim, const dim3& blockDim, const Fun& function) {
    if (device == ComputeDevice::CUDA && (allowed_devices == ComputeDevice::CUDA || allowed_devices == ComputeDevice::Both)) {
#ifdef __CUDACC__
        detail::gpe_generic_cuda_kernel<<<gridDim, blockDim>>>(function);
        #if not defined(GPE_NO_CUDA_ERROR_CHECKING) and not defined(NDEBUG)
        detail::gpu_assert(cudaPeekAtLastError());
        detail::gpu_assert(cudaDeviceSynchronize());
        #endif
#else
        std::cerr << "gpe::start_parallel with device CUDA but no CUDA support!" << std::endl;
        exit(1);
#endif
    }
    else if (device == ComputeDevice::CPU && (allowed_devices == ComputeDevice::CPU || allowed_devices == ComputeDevice::Both)) {
        detail::gpe_start_cpu_parallel(gridDim, blockDim, function);
    }
    else {
        std::cerr << "gpe::start_parallel with device CPU but no CPU kernel!" << std::endl;
        exit(1);
    }
}




} // namespace gpe


#endif // PARALLEL_START_H
