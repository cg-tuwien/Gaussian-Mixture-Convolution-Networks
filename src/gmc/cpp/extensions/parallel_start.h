#ifndef PARALLEL_START_H
#define PARALLEL_START_H

#include <cassert>
#include <iostream>
#include <omp.h>

#include <cuda_runtime.h>

#include "cuda_qt_creator_definitinos.h"


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
