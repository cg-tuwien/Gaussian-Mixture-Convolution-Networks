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

#include <vector>
#include <algorithm>

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void farthestpointsamplingLauncher2d(int b, int n, int m, const float * inp, float * temp, int * out);
void farthestpointsamplingLauncher3d(int b, int n, int m, const float * inp, float * temp, int * out);

torch::Tensor farthest_point_sampling(torch::Tensor points, int n_samples) {
    const auto n_dims = points.size(-1);
    const auto n_points = points.size(-2);
    const auto original_shape = points.sizes().vec();
    points = points.view({-1, n_points, n_dims}).contiguous();
    const auto n_batch = points.size(0);
    
    auto tempData = torch::empty({n_points * 32}, torch::TensorOptions(points.device()).dtype(torch::ScalarType::Float));
    auto indices_out = torch::empty({n_batch, n_samples}, torch::TensorOptions(points.device()).dtype(torch::ScalarType::Int));

    if (n_dims == 2) {
        farthestpointsamplingLauncher2d(n_batch, n_points, n_samples, points.data_ptr<float>(), tempData.data_ptr<float>(), indices_out.data_ptr<int>());
    }
    else {
        farthestpointsamplingLauncher3d(n_batch, n_points, n_samples, points.data_ptr<float>(), tempData.data_ptr<float>(), indices_out.data_ptr<int>());
    }
    
    auto out_shape = original_shape;
    out_shape.pop_back();
    out_shape.back() = n_samples;
    return indices_out.view(out_shape);
}

#ifndef GMC_CMAKE_TEST_BUILD
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("apply", &farthest_point_sampling, "farthest_point_sampling (CUDA)");
//  m.def("backward", &cuda_parallel_backward, "evaluate_inversed backward (CUDA)");
}
#endif

// class FarthestPointSampleGpuOp: public OpKernel{
//   public:
//     explicit FarthestPointSampleGpuOp(OpKernelConstruction* context):OpKernel(context) {
//                     OP_REQUIRES_OK(context, context->GetAttr("npoint", &npoint_));
//                     OP_REQUIRES(context, npoint_ > 0, errors::InvalidArgument("FarthestPointSample expects positive npoint"));
//                 }
//     void Compute(OpKernelContext * context)override{
//       int m = npoint_;
// 
//       const Tensor& inp_tensor=context->input(0);
//       OP_REQUIRES(context,inp_tensor.dims()==3 && inp_tensor.shape().dim_size(2)==3,errors::InvalidArgument("FarthestPointSample expects (batch_size,num_points,3) inp shape"));
//       int b=inp_tensor.shape().dim_size(0);
//       int n=inp_tensor.shape().dim_size(1);
//       auto inp_flat=inp_tensor.flat<float>();
//       const float * inp=&(inp_flat(0));
//       Tensor * out_tensor;
//       OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,m},&out_tensor));
//       auto out_flat=out_tensor->flat<int>();
//       int * out=&(out_flat(0));
//       Tensor temp_tensor;
//       OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<float>::value,TensorShape{32,n},&temp_tensor));
//       auto temp_flat=temp_tensor.flat<float>();
//       float * temp=&(temp_flat(0));
//       farthestpointsamplingLauncher(b,n,m,inp,temp,out);
//     }
//     private:
//         int npoint_;
// };
// REGISTER_KERNEL_BUILDER(Name("FarthestPointSample").Device(DEVICE_GPU),FarthestPointSampleGpuOp);

