from torch.utils.cpp_extension import load
import os

print("abrakadabra")

source_dir = os.path.dirname(__file__)
print(source_dir)

#gm_evaluate_inversed_cuda = load(
    #'gm_evaluate_inversed_cuda', ['gm_evaluate_inversed_cuda.cpp', 'gm_evaluate_inversed_cuda.cu'], verbose=True)
#help(gm_evaluate_inversed_cuda)

extra_include_paths = [source_dir + "/../glm/"]

gm_evaluate_inversed_cpu = load('gm_evaluate_inversed_cpu', [source_dir + '/gm_evaluate_inversed_cpu.cpp'],
                                extra_include_paths=extra_include_paths,
                                verbose=True, extra_cflags=["-fopenmp", "-O4", "-ffast-math"], extra_ldflags=["-lpthread"])

# help(gm_evaluate_inversed_cpu)
