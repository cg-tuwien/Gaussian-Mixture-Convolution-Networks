#ifndef CUDA_QT_CREATOR_DEFINITINOS_H
#define CUDA_QT_CREATOR_DEFINITINOS_H

#include <cuda_runtime.h>
#include <algorithm>

#ifndef __CUDACC__

constexpr dim3 gridDim;
constexpr dim3 blockDim;
constexpr dim3 blockIdx;
constexpr dim3 threadIdx;

using std::min;
using std::max;

#endif

#endif // CUDA_QT_CREATOR_DEFINITINOS_H
