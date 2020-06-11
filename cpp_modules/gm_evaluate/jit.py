from torch.utils.cpp_extension import load
import os

print("abrakadabra")

source_dir = os.path.dirname(__file__)
print(source_dir)

#gm_evaluate_inversed_cuda = load(
    #'gm_evaluate_inversed_cuda', ['gm_evaluate_inversed_cuda.cpp', 'gm_evaluate_inversed_cuda.cu'], verbose=True)
#help(gm_evaluate_inversed_cuda)

gm_evaluate_inversed_cpu = load(
   'gm_evaluate_inversed_cpu', [source_dir + '/gm_evaluate_inversed_cpu.cpp'], verbose=True)
help(gm_evaluate_inversed_cpu)
