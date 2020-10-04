/* Furthest point sampling
 * Original author: Haoqiang Fan
 * Copyright (C) 2017 Charles R. Qi
 * Copyright (C) 2020 Adam Celarek, Research Unit of Computer Graphics, TU Wien
 *
 * taken from https://github.com/charlesq34/pointnet2/tree/master/tf_ops/sampling

The MIT License (MIT)

Copyright (c) 2017 Charles R. Qi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "../cuda_qt_creator_definitinos.h"
#include "../math/matrix.h"
#include <glm/glm.hpp>

template <int N_DIMS>
__global__ void farthestpointsamplingKernel(int b, int n, int m,
                                            const float *dataset, float *temp,
                                            int *idxs) {
    using Vec = glm::vec<N_DIMS, float>;
    if (m <= 0)
        return;
    const int BlockSize = 512;
    __shared__ float dists[BlockSize];
    __shared__ int dists_i[BlockSize];
    const int BufferSize = 3072;
    __shared__ float buf[BufferSize * N_DIMS];
    for (int i = blockIdx.x; i < b; i += gridDim.x) {
        int old = 0;
        if (threadIdx.x == 0)
            idxs[i * m + 0] = old;
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            temp[blockIdx.x * n + j] = 1e38;
        }
        for (int j = threadIdx.x; j < min(BufferSize, n) * N_DIMS;
             j += blockDim.x) {
            buf[j] = dataset[i * n * N_DIMS + j];
        }
        __syncthreads();
        for (int j = 1; j < m; j++) {
            int besti = 0;
            float best = -1;
            Vec point1 =
                *reinterpret_cast<const Vec *>(&dataset[i * n * N_DIMS + old * N_DIMS]);
            for (int k = threadIdx.x; k < n; k += blockDim.x) {
                float td = temp[blockIdx.x * n + k];
                Vec point2;
                if (k < BufferSize) {
                    point2 = *reinterpret_cast<Vec *>(&buf[k * N_DIMS]);
                } else {
                    point2 =
                        *reinterpret_cast<const Vec *>(&dataset[i * n * N_DIMS + k * N_DIMS]);
                }
                float d = gpe::squared_norm(point2 - point1);
                float d2 = min(d, td);
                if (d2 != td)
                    temp[blockIdx.x * n + k] = d2;
                if (d2 > best) {
                    best = d2;
                    besti = k;
                }
            }
            dists[threadIdx.x] = best;
            dists_i[threadIdx.x] = besti;
            for (int u = 0; (1 << u) < blockDim.x; u++) {
                __syncthreads();
                if (threadIdx.x < (blockDim.x >> (u + 1))) {
                    int i1 = (threadIdx.x * 2) << u;
                    int i2 = (threadIdx.x * 2 + 1) << u;
                    if (dists[i1] < dists[i2]) {
                        dists[i1] = dists[i2];
                        dists_i[i1] = dists_i[i2];
                    }
                }
            }
            __syncthreads();
            old = dists_i[0];
            if (threadIdx.x == 0)
                idxs[i * m + j] = old;
        }
    }
}

//require 32*n working space
void farthestpointsamplingLauncher2d(int b,int n,int m,const float * inp,float * temp,int * out){
    farthestpointsamplingKernel<2><<<32,512>>>(b,n,m,inp,temp,out);
}
void farthestpointsamplingLauncher3d(int b,int n,int m,const float * inp,float * temp,int * out){
    farthestpointsamplingKernel<3><<<32,512>>>(b,n,m,inp,temp,out);
}
