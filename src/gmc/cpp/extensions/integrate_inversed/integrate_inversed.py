from torch.utils.cpp_extension import load
import os
import torch.autograd
from gmc.cpp.extensions.compile_flags import *

source_dir = os.path.dirname(__file__)
print(source_dir)

extra_include_paths = [source_dir + "/../../glm/", source_dir + "/../../cub/", source_dir + "/.."]

# cuda = load('evaluate_inversed_cuda_parallel', [source_dir + '/cuda_parallel.cpp', source_dir + '/cuda_parallel.cu'],
#                                 extra_include_paths=extra_include_paths,
#                                 verbose=True, extra_cflags=cuda_extra_cflags, extra_cuda_cflags=cuda_extra_cuda_cflags, extra_ldflags=["-lpthread"])

#cuda_bvh = load('evaluate_inversed_cuda_bvh', [source_dir + '/cuda_bvh.cpp', source_dir + '/cuda_bvh.cu', source_dir + '/../lbvh/bvh.cu',
#                                               source_dir + '/../math/symeig_cuda.cpp', source_dir + '/../math/symeig_cuda.cu'],
#                                extra_include_paths=extra_include_paths,
#                                verbose=True, extra_cflags=cuda_extra_cflags, extra_cuda_cflags=cuda_extra_cuda_cflags, extra_ldflags=["-lpthread"])

parallel = load('integrate_inversed_parallel',
                [source_dir + '/parallel_binding.cpp',
                 source_dir + '/parallel_implementation.cu'],
                extra_include_paths=extra_include_paths, verbose=True, extra_cflags=cuda_extra_cflags, extra_cuda_cflags=cuda_extra_cuda_cflags, extra_ldflags=["-lpthread"])

# cpu = load('evaluate_inversed_cpu_parallel', [source_dir + '/cpu_parallel.cpp'],
#                                 extra_include_paths=extra_include_paths,
#                                 verbose=True, extra_cflags=cpp_extra_cflags, extra_ldflags=["-lpthread"])

class IntegrateInversed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mixture: torch.Tensor):
        if not mixture.is_contiguous():
            mixture = mixture.contiguous()

        output = parallel.forward(mixture)
        ctx.save_for_backward(mixture)

        return output

    #@staticmethod
    #def backward(ctx, grad_output):
        #if not grad_output.is_contiguous():
            #grad_output = grad_output.contiguous()

        #mixture = ctx.saved_tensors
        #grad_mixture = parallel.backward(grad_output, mixture, ctx.needs_input_grad[0])

        #return grad_mixture


apply = IntegrateInversed.apply
