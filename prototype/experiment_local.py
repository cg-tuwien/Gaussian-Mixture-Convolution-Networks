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
import gm_evaluate.gm_evaluate_inversed
import gm
import torch.autograd

mixture = gm.generate_random_mixtures(50, 5, 1200, 3)
mixture = gm.pack_mixture(gm.weights(mixture), gm.positions(mixture), gm.covariances(mixture).inverse().transpose(-2, -1))
xes = torch.rand([50, 5, 300, 3])

ref = gm.evaluate_inversed(mixture, xes)
start_time = time.perf_counter()
print("python started")
ref = gm.evaluate_inversed(mixture, xes)
ref_time = time.perf_counter()
print("cpp started")
out = gm_evaluate.gm_evaluate_inversed.apply(mixture, xes)
cpp_time = time.perf_counter()

print(f"====== requires_grad = False ======")
print(f"RMSE: {((ref - out)**2).mean().sqrt().item()}")
print(f"python: {ref_time - start_time}")
print(f"cpp: {cpp_time - ref_time}")

print(f"====== requires_grad = True ======")
mixture.requires_grad = True;
xes.requires_grad = True;

print("cpu forward started")
cpu_start = time.perf_counter()
out = gm_evaluate.gm_evaluate_inversed.apply(mixture, xes)
cpu_forward = time.perf_counter()
print("cpu backward started")
out.sum().backward()
cpu_backward = time.perf_counter()
cpu_mixture_grad = mixture.grad.clone()
cpu_xes_grad = xes.grad.clone()

xes.grad = None
mixture.grad = None

print("python forward started")
python_start = time.perf_counter()
ref = gm.evaluate_inversed(mixture, xes)
python_forward = time.perf_counter()
print("python backward started")
ref.sum().backward()
python_backward = time.perf_counter()
pyhton_mixture_grad = mixture.grad.clone()
pyhton_xes_grad = xes.grad.clone()


print(f"mixture grad RMSE: {((pyhton_mixture_grad - cpu_mixture_grad)**2).mean().sqrt().item()}")
print(f"xes grad RMSE: {((pyhton_xes_grad - cpu_xes_grad)**2).mean().sqrt().item()}")
print(f"python forward: {python_forward - python_start}")
print(f"python backward: {python_backward - python_forward}")
print(f"cpu forward: {cpu_forward - cpu_start}")
print(f"cpu backward: {cpu_backward - cpu_forward}")

# gradcheck takes a tuple of tensors as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.

mixture = gm.generate_random_mixtures(1, 1, 60, 3).to(torch.float64)
xes = torch.rand([1, 1, 60, 3]).to(torch.float64)

mixture.requires_grad = False;
xes.requires_grad = True;

print(f"====== torch.autograd.gradcheck ======")
test = torch.autograd.gradcheck(gm_evaluate.gm_evaluate_inversed.apply, (mixture, xes), eps=1e-6, atol=1e-4)
print(test)
