from experiment_dl_conf import *

# experiment_gm_mnist.experiment_alternating(device=list(sys.argv)[1], n_epochs=200, n_epochs_fitting_training=2, desc_string="_pS_lrnAmp",
#                                              kernel_learning_rate=learning_rate_kernels, fitting_learning_rate=learning_rate_fitting,
#                                              layer1_m2m_fitting=generate_fitting_module_S,
#                                              layer2_m2m_fitting=generate_fitting_module_S,
#                                              layer3_m2m_fitting=generate_fitting_module_S,
#                                              learn_covariances_after=200, learn_positions_after=200,
#                                              log_interval=log_interval)

import os
source_dir = os.path.dirname(__file__)
sys.path.append(source_dir + '/../cpp_modules')

import time

import torch
import gm_evaluate.jit
import gm

mixture = gm.generate_random_mixtures(50, 10, 600, 2)
xes = torch.rand([50, 10, 600, 2])

start_time = time.perf_counter()
print("python started")
ref = gm.evaluate_inversed(mixture, xes)
ref_time = time.perf_counter()
print("cpp started")
out = gm_evaluate.jit.gm_evaluate_inversed_cpu.forward(mixture, xes)
cpp_time = time.perf_counter()


print(f"err: {((ref - out)**2).mean().item()}")
print(f"python: {ref_time - start_time}")
print(f"cpp: {cpp_time - ref_time}")
