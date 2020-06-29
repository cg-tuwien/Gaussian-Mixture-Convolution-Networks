from torch.utils.cpp_extension import load
import os
import torch.autograd

source_dir = os.path.dirname(__file__)
print(source_dir)



extra_include_paths = [source_dir + "/../../glm/", source_dir + "/.."]

#cuda = load('evaluate_inversed_cuda', [source_dir + '/evaluate_inversed_cuda.cpp', source_dir + '/evaluate_inversed_cuda.cu'],
                                #extra_include_paths=extra_include_paths,
                                #verbose=True, extra_cflags=["-O4", "-ffast-math"], extra_cuda_cflags=["-O3",  "--use_fast_math"])
cpu = load('em_fitting_cpu', [source_dir + '/em_fitting_cpu.cpp'],
                                extra_include_paths=extra_include_paths,
                                verbose=True, extra_cflags=["-fopenmp", "-O4", "-ffast-math"], extra_ldflags=["-lpthread"])

class EmFitting(torch.autograd.Function):
    @staticmethod
    def forward(ctx, target: torch.Tensor, initial: torch.Tensor):
        if not target.is_contiguous():
            target = target.contiguous()

        if not initial.is_contiguous():
            initial = initial.contiguous()

        if target.is_cuda:
            assert False
            #output = cuda.forward(mixture, xes)
        else:
            output, assignments = cpu.forward(target, initial)
        ctx.save_for_backward(target, assignments)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert False
        #if not grad_output.is_contiguous():
            #grad_output = grad_output.contiguous()

        #mixture, xes = ctx.saved_tensors
        #if mixture.is_cuda:
            #grad_mixture, grad_xes = cuda.backward(grad_output, mixture, xes, ctx.needs_input_grad[0], ctx.needs_input_grad[1])
        #else:
            #grad_mixture, grad_xes = cpu.backward(grad_output, mixture, xes, ctx.needs_input_grad[0], ctx.needs_input_grad[1])
        #return grad_mixture, grad_xes


apply = EmFitting.apply
