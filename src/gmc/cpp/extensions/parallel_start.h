#ifndef PARALLEL_START_H
#define PARALLEL_START_H

#include <cassert>
#include <iostream>
#include <omp.h>

#include <torch/all.h>
#include <cuda_runtime.h>

#include "cuda_qt_creator_definitinos.h"

namespace gpe {

template <class T>
__host__ __device__ __forceinline__ T atomicAdd(T *ptr, T val) {
//    *ptr += val;
//    return *ptr;
#ifdef __CUDA_ARCH__
    return ::atomicAdd(ptr, val);
#elif defined(_OPENMP) and _OPENMP>=201107
    T t;
    #pragma omp atomic capture
    { t = *ptr; *ptr += val; }
    return t;
#else
    #error "Requires gcc or OpenMP>=3.1"
#endif
}

template <typename T>
struct RestrictPtrTraits {
    typedef T* __restrict__ PtrType;
};

namespace detail {

//void gpe_start_cpu_parallel(const dim3& gridDim, const dim3& blockDim, const std::function<void(const dim3&, const dim3&, const dim3&, const dim3&)>& function);
template <typename Fun>
inline void gpe_start_cpu_parallel(const dim3& gridDim, const dim3& blockDim, Fun function) {
    #pragma omp parallel for num_threads(omp_get_num_procs()) collapse(3)
    for (unsigned blockIdxZ = 0; blockIdxZ < gridDim.z; ++blockIdxZ) {
        for (unsigned blockIdxY = 0; blockIdxY < gridDim.y; ++blockIdxY) {
            for (unsigned blockIdxX = 0; blockIdxX < gridDim.x; ++blockIdxX) {
                const auto blockIdx = dim3{blockIdxX, blockIdxY, blockIdxZ};
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
    CPU, CUDA
};



inline ComputeDevice device(const torch::Tensor& t) {
    return t.is_cuda() ? ComputeDevice::CUDA : ComputeDevice::CPU;
}



template <typename Fun>
void start_parallel(ComputeDevice device, const dim3& gridDim, const dim3& blockDim, const Fun& function) {
    switch (device) {
#ifdef __CUDACC__
        case ComputeDevice::CUDA:
            detail::gpe_generic_cuda_kernel<<<gridDim, blockDim>>>(function);
            #if not defined(GPE_NO_CUDA_ERROR_CHECKING) and not defined(NDEBUG)
            detail::gpu_assert(cudaPeekAtLastError());
            detail::gpu_assert(cudaDeviceSynchronize());
            #endif
        break;
#endif
        case ComputeDevice::CPU:
        default:
            detail::gpe_start_cpu_parallel(gridDim, blockDim, function);
        break;
    }
}




} // namespace gpe


#endif // PARALLEL_START_H
