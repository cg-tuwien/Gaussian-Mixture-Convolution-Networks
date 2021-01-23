#ifndef PARALLEL_START_H
#define PARALLEL_START_H

#include <atomic>
#include <cassert>
#include <string>
#include <omp.h>

#include <cuda_runtime.h>
#include <torch/types.h>

#include "common.h"
#include "cuda_operations.h"
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

enum class ComputeDevice {
    CPU, CUDA, Both
};

namespace detail {

inline dim3 to3dIdx(const dim3& dimension, unsigned idx ) {
    unsigned x = idx / (dimension.z * dimension.y);
    idx -= x * dimension.z * dimension.y;
    unsigned y = idx / dimension.z;
    unsigned z = idx - y * dimension.z;
//    unsigned x = idx % dimension.x;
    return { x, y, z };
}

//template <typename Fun>
//inline void gpe_start_cpu_parallel(const dim3& gridDim, const dim3& blockDim, Fun function) {
//    #pragma omp parallel for num_threads(omp_get_num_procs()) collapse(3)
//    for (unsigned blockIdxZ = 0; blockIdxZ < gridDim.z; ++blockIdxZ) {
//        for (unsigned blockIdxY = 0; blockIdxY < gridDim.y; ++blockIdxY) {
//            for (unsigned blockIdxX = 0; blockIdxX < gridDim.x; ++blockIdxX) {
//                const auto blockIdx = dim3{blockIdxX, blockIdxY, blockIdxZ};
//                dim3 threadIdx = {0, 0, 0};
//                for (threadIdx.z = 0; threadIdx.z < blockDim.z; ++threadIdx.z) {
//                    for (threadIdx.y = 0; threadIdx.y < blockDim.y; ++threadIdx.y) {
//                        for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) {
//                            function(gridDim, blockDim, blockIdx, threadIdx);
//                        }
//                    }
//                }
////                const auto n = blockDim.x * blockDim.y * blockDim.z;
////                #pragma omp parallel for num_threads(std::min(n, 32u))
////                for (unsigned i = 0; i < n; ++i) {
////                    dim3 threadIdx = to3dIdx(blockDim, i);
////                    function(gridDim, blockDim, blockIdx, threadIdx);
////                }
//            }
//        }
//    }
//}

template <typename Fun>
inline void gpe_start_cpu_parallel(const dim3& gridDim, const dim3& blockDim, Fun function) {
    const auto n = blockDim.x * blockDim.y * blockDim.z;
    const auto thread_count = n;//std::min(n, 64u);

    gpe::detail::CpuSynchronisationPoint::setThreadCount(thread_count);

    #pragma omp parallel for num_threads(thread_count)
    for (int i = 0; i < n; ++i) {
        dim3 threadIdx = to3dIdx(blockDim, i);
        for (unsigned blockIdxZ = 0; blockIdxZ < gridDim.z; ++blockIdxZ) {
            for (unsigned blockIdxY = 0; blockIdxY < gridDim.y; ++blockIdxY) {
                for (unsigned blockIdxX = 0; blockIdxX < gridDim.x; ++blockIdxX) {
                    const auto blockIdx = dim3{blockIdxX, blockIdxY, blockIdxZ};
                    function(gridDim, blockDim, blockIdx, threadIdx);
                }
            }
        }
    }
}

template <typename Fun>
inline void gpe_start_cpu_serial(const dim3& gridDim, const dim3& blockDim, Fun function) {
    const auto n = blockDim.x * blockDim.y * blockDim.z;

    gpe::detail::CpuSynchronisationPoint::setThreadCount(1);

    for (unsigned i = 0; i < n; ++i) {
        dim3 threadIdx = to3dIdx(blockDim, i);
        for (unsigned blockIdxZ = 0; blockIdxZ < gridDim.z; ++blockIdxZ) {
            for (unsigned blockIdxY = 0; blockIdxY < gridDim.y; ++blockIdxY) {
                for (unsigned blockIdxX = 0; blockIdxX < gridDim.x; ++blockIdxX) {
                    const auto blockIdx = dim3{blockIdxX, blockIdxY, blockIdxZ};
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

inline void gpu_assert(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert in start_parallel: %s\n", cudaGetErrorString(code));
        exit(code);
    }
}
#endif

template <ComputeDevice device, typename Fun>
struct CudaStarter {
    void operator()(const dim3& gridDim, const dim3& blockDim, const Fun& function) {
        GPE_UNUSED(gridDim)
        GPE_UNUSED(blockDim)
        GPE_UNUSED(function)
#ifdef __CUDACC__
        detail::gpe_generic_cuda_kernel<<<gridDim, blockDim>>>(function);
        detail::gpu_assert(cudaPeekAtLastError());
        detail::gpu_assert(cudaDeviceSynchronize());
#else
        std::cerr << "gpe::start_parallel with device CUDA but no CUDA support!" << std::endl;
        exit(1);
#endif
    }
};

template <typename Fun>
struct CudaStarter<ComputeDevice::CPU, Fun> {
    void operator()(const dim3& gridDim, const dim3& blockDim, const Fun& function) {
        GPE_UNUSED(gridDim)
        GPE_UNUSED(blockDim)
        GPE_UNUSED(function)
        std::cerr << "gpe::start_parallel with device CUDA but CUDA device not allowed!" << std::endl;
        exit(1);
    }
};

//struct Cuda
} // namespace detail

inline ComputeDevice device(const torch::Tensor& t) {
    return t.is_cuda() ? ComputeDevice::CUDA : ComputeDevice::CPU;
}

template <ComputeDevice allowed_devices, typename Fun>
void start_parallel(ComputeDevice device, const dim3& gridDim, const dim3& blockDim, const Fun& function) {
    if (device == ComputeDevice::CUDA) {
        detail::CudaStarter<allowed_devices, Fun> ()(gridDim, blockDim, function);
    }
    else if (device == ComputeDevice::CPU && (allowed_devices == ComputeDevice::CPU || allowed_devices == ComputeDevice::Both)) {
        detail::gpe_start_cpu_parallel(gridDim, blockDim, function);
    }
    else {
        std::cerr << "gpe::start_parallel with device CPU but no CPU kernel!" << std::endl;
        exit(1);
    }
}

template <typename Fun>
void start_serial(ComputeDevice device, const dim3& gridDim, const dim3& blockDim, const Fun& function) {
    if (device == ComputeDevice::CPU) {
        detail::gpe_start_cpu_serial(gridDim, blockDim, function);
    }
    else {
        std::cerr << "gpe::start_serial with device GPU not allowed!" << std::endl;
        exit(1);
    }
}



} // namespace gpe


#endif // PARALLEL_START_H
