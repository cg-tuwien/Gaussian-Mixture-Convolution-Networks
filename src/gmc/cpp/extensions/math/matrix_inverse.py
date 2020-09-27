from torch.utils.cpp_extension import load
import os
import torch.autograd
import platform

source_dir = os.path.dirname(__file__)
print(source_dir)

extra_include_paths = [source_dir + "/../../glm/", source_dir + "/.."]

cuda_extra_cuda_cflags = ["-O3",  "--use_fast_math", "--std=c++14", "--expt-extended-lambda", "--default-stream per-thread"]
if platform.system() == "Windows":
    cuda_extra_cflags = ["/O2", "/fp:fast", "/std:c++14"]
    cpp_extra_cflags = ["/openmp", "/O2", "/fp:fast", "/std:c++14"]
else:
    cuda_extra_cflags = ["-O4", "-ffast-math", "-march=native", "--std=c++14"];
    cpp_extra_cflags = ["-fopenmp", "-O4", "-ffast-math", "-march=native", "--std=c++14"]

cuda = load('matrix_inverse_cuda', [source_dir + '/matrix_inverse_cuda.cpp', source_dir + '/matrix_inverse_cuda.cu'],
            extra_include_paths=extra_include_paths,
            verbose=True, extra_cflags=cuda_extra_cflags, extra_cuda_cflags=cuda_extra_cuda_cflags)


class MatrixInverse(torch.autograd.Function):
    """
    Faster matrix inverse compared to pytorch for 2x2 and 3x3 matrices
    """
    @staticmethod
    def forward(ctx, matrices: torch.Tensor):
        if not matrices.is_contiguous():
            matrices = matrices.contiguous()

        if matrices.is_cuda:
            output = cuda.forward(matrices)
        else:
            output = matrices.inverse()
        ctx.save_for_backward(matrices, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        matrices, matrices_inverse = ctx.saved_tensors
        return -matrices_inverse @ grad_output @ matrices_inverse


apply = MatrixInverse.apply
