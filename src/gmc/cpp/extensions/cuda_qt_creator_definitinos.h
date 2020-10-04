#ifndef CUDA_QT_CREATOR_DEFINITINOS_H
#define CUDA_QT_CREATOR_DEFINITINOS_H

#include <cuda_runtime.h>

#ifndef __CUDACC__

constexpr dim3 gridDim;
constexpr dim3 blockDim;
constexpr dim3 blockIdx;
constexpr dim3 threadIdx;

int atomicCAS(int* address, int compare, int val);

#endif

#endif // CUDA_QT_CREATOR_DEFINITINOS_H
