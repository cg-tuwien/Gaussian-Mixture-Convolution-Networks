#ifndef PARALLEL_START_H
#define PARALLEL_START_H

#include <cassert>
#include <iostream>
#include <omp.h>

#include <torch/all.h>
#include <cuda_runtime.h>

//#ifndef __CUDA_ARCH__
//    #if defined(_OPENMP) and _OPENMP>=201107
//    template <class T>
//    __host__ inline T atomicAdd(T *ptr, T val) {
//        T t;
//    #pragma omp atomic capture
//        { t = *ptr; *ptr += val; }
//        return t;
//    }
//    #else
//    #error "Requires gcc or OpenMP>=3.1"
//    #endif
//#endif

namespace gpe {

template <class T>
__host__ __device__ __forceinline__ T atomicAdd(T *ptr, T val) {
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


namespace detail {

dim3 to3dIdx(const dim3& dimension, unsigned idx ) {
    unsigned x = idx / (dimension.z * dimension.y);
    idx -= x * dimension.z * dimension.y;
    unsigned y = idx / dimension.z;
    unsigned z = idx - y * dimension.z;
//    unsigned x = idx % dimension.x;
    return { x, y, z };
}

template <typename Fun>
void gpe_start_cpu_parallel(const dim3& gridDim, const dim3& blockDim, const Fun& function) {
    #pragma omp parallel for num_threads(omp_get_num_procs())
    for (unsigned grid_i = 0; grid_i < gridDim.x * gridDim.y * gridDim.z; ++grid_i) {
        const dim3 blockIdx = to3dIdx(gridDim, grid_i);
        assert(blockIdx.x >= 0 && blockIdx.x < gridDim.x);
        assert(blockIdx.y >= 0 && blockIdx.y < gridDim.y);
        assert(blockIdx.z >= 0 && blockIdx.z < gridDim.z);
        const auto num_block_threads = blockDim.x * blockDim.y * blockDim.z;
        // at least simulate a warp // can't go much higher because win supports only 64
//        const auto num_real_threads = std::min(std::max(omp_get_num_procs(), 32), int(num_block_threads));
//        std::cout << "gpe_start_cpu_parallel/num_real_threads: " << num_real_threads << std::endl;
//        #pragma omp parallel for num_threads(num_real_threads)
//        #pragma omp parallel for num_threads(omp_get_num_procs())
//        for (unsigned block_i = 0; block_i < num_block_threads; ++block_i) {
//            const dim3 threadIdx = to3dIdx(blockDim, block_i);
//            assert(threadIdx.x >= 0 && threadIdx.x < blockDim.x);
//            assert(threadIdx.y >= 0 && threadIdx.y < blockDim.y);
//            assert(threadIdx.z >= 0 && threadIdx.z < blockDim.z);
//            function(gridDim, blockDim, blockIdx, threadIdx);
//        }

        dim3 threadIdx = {0, 0, 0};
        for (; threadIdx.z < blockDim.z; ++threadIdx.z) {
            for (threadIdx.y = 0; threadIdx.y < blockDim.y; ++threadIdx.y) {
                for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) {
                    function(gridDim, blockDim, blockIdx, threadIdx);
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
#endif

} // namespace detail

enum class ComputeDevice {
    CPU, CUDA
};



ComputeDevice device(const torch::Tensor& t) {
    return t.is_cuda() ? ComputeDevice::CUDA : ComputeDevice::CPU;
}

template <typename Fun>
void start_parallel(ComputeDevice device, const dim3& gridDim, const dim3& blockDim, const Fun& function) {
    switch (device) {
#ifdef __CUDACC__
        case ComputeDevice::CUDA:
            detail::gpe_generic_cuda_kernel<<<gridDim, blockDim>>>(function);
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
