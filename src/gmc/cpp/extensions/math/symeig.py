from torch.utils.cpp_extension import load
import os

import torch.autograd

from gmc.cpp.extensions.compile_flags import *

source_dir = os.path.dirname(__file__)
print(source_dir)

extra_include_paths = [source_dir + "/../../glm/", source_dir + "/.."]

cuda = load('symeig_cuda', [source_dir + '/symeig_cuda.cpp', source_dir + '/symeig_cuda.cu'],
            extra_include_paths=extra_include_paths,
            verbose=True, extra_cflags=cuda_extra_cflags, extra_cuda_cflags=cuda_extra_cuda_cflags)
cpu = load('symeig_cpu', [source_dir + '/symeig_cpu.cpp'],
           extra_include_paths=extra_include_paths,
           verbose=True, extra_cflags=cpp_extra_cflags, extra_ldflags=["-lpthread"])


class SymEig(torch.autograd.Function):
    """
   This is only tested for symmetric positive definite 2x2 and 3x3 matrices with positive only diagonal
   """
    @staticmethod
    def forward(ctx, matrices: torch.Tensor):
        if not matrices.is_contiguous():
            matrices = matrices.contiguous()

        ctx.save_for_backward(matrices)
        if matrices.is_cuda:
            output = cuda.forward(matrices)
        else:
            output = cpu.forward(matrices)
        return tuple(output)

    # @staticmethod
    # def backward(ctx, grad_output):
    #     if not grad_output.is_contiguous():
    #         grad_output = grad_output.contiguous()
    #
    #     mixture, xes = ctx.saved_tensors
    #     if mixture.is_cuda:
    #         grad_mixture, grad_xes = cuda.backward(grad_output, mixture, xes, ctx.needs_input_grad[0], ctx.needs_input_grad[1])
    #     else:
    #         grad_mixture, grad_xes = cpu.backward(grad_output, mixture, xes, ctx.needs_input_grad[0], ctx.needs_input_grad[1])
    #     return grad_mixture, grad_xes


apply = SymEig.apply
