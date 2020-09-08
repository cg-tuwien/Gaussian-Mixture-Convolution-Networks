from torch.utils.cpp_extension import load
import os
import torch.autograd
import platform

source_dir = os.path.dirname(__file__)
print(source_dir)



extra_include_paths = [source_dir + "/../../glm/", source_dir + "/.."]

if platform.system() == "Windows":
    cuda_extra_cflags = ["/O2", "/fp:fast", "/std:c++14"]
    cuda_extra_cuda_cflags = ["-O3",  "--use_fast_math", "--std=c++14"]
    cpp_extra_cflags = ["/openmp", "/O2", "/fp:fast", "/std:c++14"]
else:
    cuda_extra_cflags = ["-O4", "-ffast-math"];
    cuda_extra_cuda_cflags = ["-arch=sm_70", "-O3", "--use_fast_math"]
    cpp_extra_cflags = ["-fopenmp", "-O4", "-ffast-math"]

# cuda = load('eigen_cuda', [source_dir + '/eigen_cuda.cpp', source_dir + '/eigen_cuda.cu'],
#                                 extra_include_paths=extra_include_paths,
#                                 verbose=True, extra_cflags=cuda_extra_cflags, extra_cuda_cflags=cuda_extra_cuda_cflags)
cpu = load('symeig_cpu', [source_dir + '/symeig_cpu.cpp'],
           extra_include_paths=extra_include_paths,
           verbose=True, extra_cflags=cpp_extra_cflags, extra_ldflags=["-lpthread"])

t = torch.rand(2, 1, 2, 2)
t = t @ torch.transpose(t, -1, -2)
t += torch.eye(2).view(1, 1, 2, 2) * 0.000001

torch_eigenvalues, torch_eigenvectors = torch.symeig(t, True)
my_eigenvalues, my_eigenvectors = cpu.forward(t)

print(my_eigenvalues.unsqueeze(-2) - (t@my_eigenvectors.transpose(-1, -2) / my_eigenvectors.transpose(-1, -2)))

t = torch.tensor([1.0, 0.5, 0.5, 1.0]).view(1, 1, 1, 2, 2)
print(t.symeig(True))
print("=====")
print(cpu.forward(t))

# class Eigen(torch.autograd.Function):
#     """
#    This is only tested for symmetric positive definite 2x2 and 3x3 matrices with positive only diagonal
#
#    """
#     @staticmethod
#     def forward(ctx, mixture: torch.Tensor, xes: torch.Tensor):
#         if not mixture.is_contiguous():
#             mixture = mixture.contiguous()
#
#         if not xes.is_contiguous():
#             xes = xes.contiguous()
#
#         ctx.save_for_backward(mixture, xes)
#         if mixture.is_cuda:
#             output = cuda.forward(mixture, xes)
#         else:
#             output = cpu.forward(mixture, xes)
#         return output
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         if not grad_output.is_contiguous():
#             grad_output = grad_output.contiguous()
#
#         mixture, xes = ctx.saved_tensors
#         if mixture.is_cuda:
#             grad_mixture, grad_xes = cuda.backward(grad_output, mixture, xes, ctx.needs_input_grad[0], ctx.needs_input_grad[1])
#         else:
#             grad_mixture, grad_xes = cpu.backward(grad_output, mixture, xes, ctx.needs_input_grad[0], ctx.needs_input_grad[1])
#         return grad_mixture, grad_xes
#
#
# apply = Eigen.apply
