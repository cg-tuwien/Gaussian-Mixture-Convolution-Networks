from torch.utils.cpp_extension import load
import os
import torch.autograd
import platform

source_dir = os.path.dirname(__file__)
print(source_dir)



extra_include_paths = [source_dir + "/../../glm/", source_dir + "/../../cub/", source_dir + "/.."]

cuda_extra_cuda_cflags = ["-O3", "--use_fast_math", "--std=c++14", "--expt-extended-lambda", "-DNDEBUG"]
if platform.system() == "Windows":
    cuda_extra_cflags = ["/O2", "/fp:fast", "/std:c++14", "/DGPE_NO_CUDA_ERROR_CHECKING"]
    cpp_extra_cflags = ["/openmp", "/O2", "/fp:fast", "/std:c++14"]
else:
    cuda_extra_cflags = ["-O4", "-ffast-math", "-march=native", "--std=c++14", "-DGPE_NO_CUDA_ERROR_CHECKING"];
    cpp_extra_cflags = ["-fopenmp", "-O4", "-ffast-math", "-march=native", "--std=c++14"]

cuda = load('evaluate_inversed_cuda_parallel', [source_dir + '/cuda_parallel.cpp', source_dir + '/cuda_parallel.cu'],
                                extra_include_paths=extra_include_paths,
                                verbose=True, extra_cflags=cuda_extra_cflags, extra_cuda_cflags=cuda_extra_cuda_cflags)

cuda_bvh = load('evaluate_inversed_cuda_bvh', [source_dir + '/cuda_bvh.cpp', source_dir + '/cuda_bvh.cu',
                                               source_dir + '/../math/symeig_cuda.cpp', source_dir + '/../math/symeig_cuda.cu'],
                                extra_include_paths=extra_include_paths,
                                verbose=True, extra_cflags=cuda_extra_cflags, extra_cuda_cflags=cuda_extra_cuda_cflags)
cpu = load('evaluate_inversed_cpu_parallel', [source_dir + '/cpu_parallel.cpp'],
                                extra_include_paths=extra_include_paths,
                                verbose=True, extra_cflags=cpp_extra_cflags, extra_ldflags=["-lpthread"])

class EvaluateInversed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mixture: torch.Tensor, xes: torch.Tensor):
        if not mixture.is_contiguous():
            mixture = mixture.contiguous()

        if not xes.is_contiguous():
            xes = xes.contiguous()

        if mixture.is_cuda:
            output, bvh_nodes, aabbs = cuda_bvh.forward(mixture, xes)
            ctx.save_for_backward(mixture, bvh_nodes, aabbs, xes)
        else:
            output = cpu.forward(mixture, xes)
            ctx.save_for_backward(mixture, xes)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        if grad_output.is_cuda:
            mixture, bvh_nodes, aabbs, xes = ctx.saved_tensors
            grad_mixture, grad_xes = cuda_bvh.backward(grad_output, mixture, bvh_nodes, aabbs, xes, ctx.needs_input_grad[0], ctx.needs_input_grad[1])
        else:
            mixture, xes = ctx.saved_tensors
            grad_mixture, grad_xes = cpu.backward(grad_output, mixture, xes, ctx.needs_input_grad[0], ctx.needs_input_grad[1])
        return grad_mixture, grad_xes


apply = EvaluateInversed.apply
