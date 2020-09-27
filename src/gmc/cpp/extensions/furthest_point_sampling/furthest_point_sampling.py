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

cuda = load('furthest_point_sampling_cuda', [source_dir + '/furthest_point_sampling.cpp', source_dir + '/furthest_point_sampling.cu'],
            extra_include_paths=extra_include_paths,
            verbose=True, extra_cflags=cuda_extra_cflags, extra_cuda_cflags=cuda_extra_cuda_cflags)
#cpu = load('symeig_cpu', [source_dir + '/symeig_cpu.cpp'],
           #extra_include_paths=extra_include_paths,
           #verbose=True, extra_cflags=cpp_extra_cflags, extra_ldflags=["-lpthread"])

def apply(points: torch.Tensor, n_sample_points: int) -> torch.Tensor:
    assert points.is_cuda
    return cuda.apply(points, n_sample_points)