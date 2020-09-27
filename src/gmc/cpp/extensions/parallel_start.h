#ifndef PARALLEL_START_H
#define PARALLEL_START_H

#include <cassert>

#include <torch/all.h>
#include <cuda_runtime.h>

namespace gpe {

#ifdef __GNUC__
template <class T>
__host__ inline T atomicAdd(T *ptr, T val) {
    return __sync_fetch_and_add(ptr, val);
}
#elif defined(_OPENMP) and _OPENMP>=201107
template <class T>
__host__ inline T atomicAdd(T *ptr, T val) {
    T t;
    #pragma omp atomic capture
    { t = *ptr; *ptr += val; }
    return t;
}
#else
    #error "Requires gcc or OpenMP>=3.1"
#endif

#ifdef __CUDACC__
template <class T>
__device__ __forceinline__ T atomicAdd(T* address, T val) {
    return atomicAdd(address, val);
}
#endif

namespace detail {

dim3 to3dIdx(const dim3& dimension, unsigned idx ) {
    unsigned z = idx / (dimension.x * dimension.y);
    idx -= (z * dimension.x * dimension.y);
    unsigned y = idx / dimension.x;
    unsigned x = idx % dimension.x;
    return { x, y, z };
}

template <typename Fun>
void gpe_start_cpu_parallel(const dim3& gridDim, const dim3& blockDim, const Fun& function) {
    for (unsigned grid_i = 0; grid_i < gridDim.x * gridDim.y * gridDim.z; ++grid_i) {
        dim3 blockIdx = to3dIdx(gridDim, grid_i);
        #pragma omp parallel for num_threads(omp_get_num_procs())
        for (unsigned block_i = 0; block_i < blockDim.x * blockDim.y * blockDim.z; ++block_i) {
            dim3 threadIdx = to3dIdx(blockDim, block_i);
            function(gridDim, blockDim, blockIdx, threadIdx);
        }
    }

}

#ifdef __CUDACC__
template <typename Fun>
__global__ void gpe_generic_cuda_kernel(Fun function) {
    function(gridDim, blockDim, blockIdx, threadIdx);
}

template <typename Fun>
void gpe_start_cuda_parallel(const dim3& gridDim, Fconst dim3& blockDim, const Fun& function) {
    gpe_generic_cuda_kernel<<<gridDim, blockDim>>>(function);
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
            detail::gpe_start_cuda_parallel(gridDim, blockDim, function);
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
