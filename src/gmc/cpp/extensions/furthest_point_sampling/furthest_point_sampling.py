from torch.utils.cpp_extension import load
import os
import torch.autograd
from gmc.cpp.extensions.compile_flags import *

source_dir = os.path.dirname(__file__)
# print(source_dir)

cuda = load('furthest_point_sampling_cuda', [source_dir + '/furthest_point_sampling.cpp', source_dir + '/furthest_point_sampling.cu'],
            extra_include_paths=extra_include_paths,
            verbose=True, extra_cflags=cuda_extra_cflags, extra_cuda_cflags=cuda_extra_cuda_cflags)
#cpu = load('symeig_cpu', [source_dir + '/symeig_cpu.cpp'],
           #extra_include_paths=extra_include_paths,
           #verbose=True, extra_cflags=cpp_extra_cflags, extra_ldflags=["-lpthread"])

def apply(points: torch.Tensor, n_sample_points: int) -> torch.Tensor:
    if points.is_cuda:
        return cuda.apply(points, n_sample_points)
    else:
        return cuda.apply(points.cuda(), n_sample_points).cpu()