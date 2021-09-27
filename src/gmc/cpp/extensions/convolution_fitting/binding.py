from torch.utils.cpp_extension import load
import os
import torch.autograd

from gmc.cpp.extensions.compile_flags import *

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

        fitting, cached_pos_cov, nodesobjs, fitting_subtrees = cpp_binding.forward(data, kernels, n_components_fitting)
        ctx.save_for_backward(data, kernels, torch.tensor(n_components_fitting), fitting, cached_pos_cov, nodesobjs, fitting_subtrees )
        return fitting

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        data, kernels, n_components_fitting, fitting, cached_pos_cov, nodesobjs, fitting_subtrees = ctx.saved_tensors
        grad_data, grad_kernel = cpp_binding.backward(grad_output, data, kernels, n_components_fitting.item(), fitting, cached_pos_cov, nodesobjs, fitting_subtrees)

        return grad_data, grad_kernel, None


apply = ConvolutionFitting.apply
