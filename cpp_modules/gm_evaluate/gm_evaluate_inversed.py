from torch.utils.cpp_extension import load
import os
import torch.autograd

source_dir = os.path.dirname(__file__)
print(source_dir)



extra_include_paths = [source_dir + "/../glm/"]

cuda = load('gm_evaluate_inversed_cuda', [source_dir + '/gm_evaluate_inversed_cuda.cpp', source_dir + '/gm_evaluate_inversed_cuda.cu'],
                                extra_include_paths=extra_include_paths,
                                verbose=True, extra_cflags=["-O4", "-ffast-math"], extra_cuda_cflags=["-O3",  "--use_fast_math"])
cpu = load('gm_evaluate_inversed_cpu', [source_dir + '/gm_evaluate_inversed_cpu.cpp'],
                                extra_include_paths=extra_include_paths,
                                verbose=True, extra_cflags=["-fopenmp", "-O4", "-ffast-math"], extra_ldflags=["-lpthread"])

class EvaluateInversed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mixture: torch.Tensor, xes: torch.Tensor):
        if not mixture.is_contiguous():
            mixture = mixture.contiguous()

        if not xes.is_contiguous():
            xes = xes.contiguous()

        ctx.save_for_backward(mixture, xes)
        if mixture.is_cuda:
            output = cuda.forward(mixture, xes)
        else:
            output = cpu.forward(mixture, xes)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        mixture, xes = ctx.saved_tensors
        if mixture.is_cuda:
            grad_mixture, grad_xes = cuda.backward(grad_output, mixture, xes, ctx.needs_input_grad[0], ctx.needs_input_grad[1])
        else:
            grad_mixture, grad_xes = cpu.backward(grad_output, mixture, xes, ctx.needs_input_grad[0], ctx.needs_input_grad[1])
        return grad_mixture, grad_xes


apply = EvaluateInversed.apply
