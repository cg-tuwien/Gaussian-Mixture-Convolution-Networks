from torch.utils.cpp_extension import load
import os
import torch.autograd
import torch.linalg

from gmc.cpp.extensions.compile_flags import *

source_dir = os.path.dirname(__file__)


source_files = [source_dir + '/pieces_bindings.cpp', source_dir + '/matrix_inverse.cu', source_dir + '/pieces.cpp', source_dir + '/symeig.cu', source_dir + '/../CpuSynchronisationPoint.cpp']

pieces_binding = load('pieces_bindings', source_files,
                   extra_include_paths=extra_include_paths,
                   verbose=True, extra_cflags=cuda_extra_cflags, extra_cuda_cflags=cuda_extra_cuda_cflags, extra_ldflags=["-lpthread"])


class SymEig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrices: torch.Tensor):
        if not matrices.is_contiguous():
            matrices = matrices.contiguous()

        eigvals, eigvecs = pieces_binding.symeig(matrices)
        ctx.save_for_backward(matrices, eigvals, eigvecs)
        return eigvals, eigvecs

    @staticmethod
    def backward(ctx, grad_eigvals, grad_eigvecs):
        matrices, eigvals, eigvecs = ctx.saved_tensors
        return pieces_binding.symeig_backward(matrices, eigvals, eigvecs, grad_eigvals, grad_eigvecs)


symeig_apply = SymEig.apply

