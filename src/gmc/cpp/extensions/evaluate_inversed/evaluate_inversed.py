from torch.utils.cpp_extension import load
import os
import torch.autograd
from gmc.cpp.extensions.compile_flags import *

source_dir = os.path.dirname(__file__)
# print(source_dir)

bindings = load('evaluate_inversed',
                [source_dir + '/cuda_bvh_implementation.cu',
                 source_dir + '/evaluate_inversed_bindings.cpp',
                 source_dir + '/parallel_implementation.cu',
                 source_dir + '/parallel_implementation_optimised_backward.cu',
                 source_dir + '/parallel_implementation_optimised_forward.cu',
                 source_dir + '/../CpuSynchronisationPoint.cpp',
                 source_dir + '/../lbvh/bvh.cu'],
                extra_include_paths=extra_include_paths, verbose=True, extra_cflags=cuda_extra_cflags, extra_cuda_cflags=cuda_extra_cuda_cflags, extra_ldflags=["-lpthread"])


class EvaluateInversed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mixture: torch.Tensor, xes: torch.Tensor):
        if not mixture.is_contiguous():
            mixture = mixture.contiguous()

        if not xes.is_contiguous():
            xes = xes.contiguous()

        # if mixture.is_cuda:
        #     output, bvh_nodes, aabbs = cuda_bvh.forward(mixture, xes)
        #     ctx.save_for_backward(mixture, bvh_nodes, aabbs, xes)
        #     # output = cuda.forward(mixture, xes)
        #     # ctx.save_for_backward(mixture, xes)
        # else:
        #     output = parallel.forward(mixture, xes)
        #     ctx.save_for_backward(mixture, xes)

        output = bindings.parallel_forward(mixture, xes)
        ctx.save_for_backward(mixture, xes, *output)

        return output[0]

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        # if grad_output.is_cuda:
        #     mixture, bvh_nodes, aabbs, xes = ctx.saved_tensors
        #     grad_mixture, grad_xes = cuda_bvh.backward(grad_output, mixture, bvh_nodes, aabbs, xes, ctx.needs_input_grad[0], ctx.needs_input_grad[1])
        #     # mixture, xes = ctx.saved_tensors
        #     # grad_mixture, grad_xes = cuda.backward(grad_output, mixture, xes, ctx.needs_input_grad[0], ctx.needs_input_grad[1])
        # else:
        #     mixture, xes = ctx.saved_tensors
        #     grad_mixture, grad_xes = parallel.backward(grad_output, mixture, xes, ctx.needs_input_grad[0], ctx.needs_input_grad[1])

        mixture, xes, *output = ctx.saved_tensors
        grad_mixture, grad_xes = bindings.parallel_backward(grad_output, mixture, xes, output, ctx.needs_input_grad[0], ctx.needs_input_grad[1])

        return grad_mixture, grad_xes


apply = EvaluateInversed.apply
