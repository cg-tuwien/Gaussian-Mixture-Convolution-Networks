from torch.utils.cpp_extension import load
import os
import torch.autograd

source_dir = os.path.dirname(__file__)
print(source_dir)

#gm_evaluate_inversed_cuda = load(
    #'gm_evaluate_inversed_cuda', ['gm_evaluate_inversed_cuda.cpp', 'gm_evaluate_inversed_cuda.cu'], verbose=True)
#help(gm_evaluate_inversed_cuda)

extra_include_paths = [source_dir + "/../glm/"]

cpu = load('gm_evaluate_inversed_cpu', [source_dir + '/gm_evaluate_inversed_cpu.cpp'],
                                extra_include_paths=extra_include_paths,
                                verbose=True, extra_cflags=["-fopenmp", "-O4", "-ffast-math"], extra_ldflags=["-lpthread"])

class EvaluateInversed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mixture, xes):
        ctx.save_for_backward(mixture, xes)
        output = cpu.forward(mixture, xes)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mixture, xes = ctx.saved_tensors
        grad_mixture, grad_xes = cpu.backward(grad_output, mixture, xes, ctx.needs_input_grad[0], ctx.needs_input_grad[1])
        return grad_mixture, grad_xes


apply = EvaluateInversed.apply
