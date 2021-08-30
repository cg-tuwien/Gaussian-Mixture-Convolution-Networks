from torch.utils.cpp_extension import load
import os
import torch.autograd

from gmc.cpp.extensions.compile_flags import *
import gmc.inout

source_dir = os.path.dirname(__file__)


source_files = [source_dir + '/bindings.cpp', source_dir + '/implementation_dispatch.cpp', source_dir + '/Tree.cu', source_dir + '/../CpuSynchronisationPoint.cpp']
for dtype in ['float', 'double']:
    for reduction_n in [1, ]:
        for ndims in [2, 3]:
            for direction in ['forward', 'backward']:
                source_files.append(source_dir + f"/implementation_{direction}_instances/template_instance_implementation_{direction}_{reduction_n}_{dtype}_{ndims}.cu")

cpp_binding = load('convolution_fitting', source_files,
                   extra_include_paths=extra_include_paths,
                   verbose=True, extra_cflags=cuda_extra_cflags, extra_cuda_cflags=cuda_extra_cuda_cflags, extra_ldflags=["-lpthread"])


class ConvolutionFitting(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data: torch.Tensor, kernels: torch.Tensor, n_components_fitting: int):
        if not data.is_contiguous():
            data = data.contiguous()
        if not kernels.is_contiguous():
            kernels = kernels.contiguous()

        result = cpp_binding.forward(data, kernels, n_components_fitting)
        ctx.save_for_backward(*result, torch.tensor(n_components_fitting))
        return result[0]

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        result = ctx.saved_tensors
        grad_data, grad_kernel = cpp_binding.backward(grad_output, *result)

        if torch.any(torch.isnan(grad_data)):
            print(f"grad_target_mixture: {grad_data}")
            print(f"target_mixture has nans: {torch.any(torch.isnan(grad_data)).item()}")
            print(f"grad_output has nans: {torch.any(torch.isnan(grad_output)).item()}")
            print(f"number of nans in torch.isnan(grad_target_mixture): {torch.isnan(grad_data).sum(dim=-1).sum(dim=-1)}")
            print(f"grad_target_mixture.shape = {grad_data.shape}")
            print(f"target_mixture.shape = {grad_data.shape}")
            print(f"grad_output.shape = {grad_output.shape}")
            print(f"reduction_n = {reduction_n}")
            print(f"n_components_fitting = {n_components_fitting}")
            gmc.inout.save(grad_output, "./bad_mixture_gradient.torch")
            gmc.inout.save(grad_data, "./bad_mixture.torch")
            print(f"ahhh")
            exit(1)

        if torch.any(torch.isnan(grad_kernel)):
            print(f"grad_target_mixture: {grad_kernel}")
            print(f"target_mixture has nans: {torch.any(torch.isnan(grad_kernel)).item()}")
            print(f"grad_output has nans: {torch.any(torch.isnan(grad_output)).item()}")
            print(f"number of nans in torch.isnan(grad_target_mixture): {torch.isnan(grad_kernel).sum(dim=-1).sum(dim=-1)}")
            print(f"grad_target_mixture.shape = {grad_kernel.shape}")
            print(f"target_mixture.shape = {grad_kernel.shape}")
            print(f"grad_output.shape = {grad_output.shape}")
            print(f"reduction_n = {reduction_n}")
            print(f"n_components_fitting = {n_components_fitting}")
            gmc.inout.save(grad_output, "./bad_mixture_gradient.torch")
            gmc.inout.save(grad_kernel, "./bad_mixture.torch")
            print(f"ahhh")
            exit(1)

        # assert not torch.any(torch.isinf(mixture))

        return grad_data, grad_kernel, None


apply = ConvolutionFitting.apply
