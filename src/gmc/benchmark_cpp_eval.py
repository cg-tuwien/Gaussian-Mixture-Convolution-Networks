import unittest
import time
import torch.autograd

# import update_syspath
import gmc.mixture as gm

enable_python = True
enable_output = True

n_batch = 100
n_layers = 10
n_dims = 2
# mixture = gm.generate_random_mixtures(n_batch, n_layers, 125, n_dims)
mixture = gm.load(f"fitting_input/fitting_input_batch{0}_netlayer{0}")[0]
mixture = gm.pack_mixture(gm.weights(mixture), gm.positions(mixture), gm.covariances(mixture).inverse().transpose(-2, -1))
xes = gm.positions(mixture)
# xes = torch.rand([100, 10, 125, n_dims])

cuda_mixture = mixture.cuda()
cuda_xes = xes.cuda()

mixture.requires_grad = False;
xes.requires_grad = False;
cuda_mixture.requires_grad = False;
cuda_xes.requires_grad = False;

if enable_python:
    python_cpu = gm.old_evaluate_inversed(mixture, xes)
    python_cuda = gm.old_evaluate_inversed(cuda_mixture, cuda_xes)
cpp_cpu = gm.evaluate_inversed(mixture, xes)
cpp_cuda = gm.evaluate_inversed(cuda_mixture, cuda_xes)
torch.cuda.synchronize()

if enable_python:
    print("python cpu started")
    python_cpu_start_time = time.perf_counter()
    python_cpu = gm.old_evaluate_inversed(mixture, xes)
    python_cpu_end_time = time.perf_counter()

    print("python cuda started")
    python_cuda_start_time = time.perf_counter()
    python_cuda = gm.old_evaluate_inversed(cuda_mixture, cuda_xes)
    torch.cuda.synchronize()
    python_cuda_end_time = time.perf_counter()

print("cpp cpu started")
cpp_cpu_start_time = time.perf_counter()
cpp_cpu = gm.evaluate_inversed(mixture, xes)
cpp_cpu_end_time = time.perf_counter()

print("cpp cuda started")
cpp_cuda_start_time = time.perf_counter()
cpp_cuda = gm.evaluate_inversed(cuda_mixture, cuda_xes)
torch.cuda.synchronize()
cpp_cuda_end_time = time.perf_counter()

print(f"====== requires_grad = False ======")
if enable_python:
    print(f"RMSE python_cpu vs python_cuda: {((python_cpu - python_cuda.cpu())**2).mean().sqrt().item()}")
    print(f"RMSE python_cpu vs cpp_cpu: {((python_cpu - cpp_cpu)**2).mean().sqrt().item()}")
    print(f"RMSE python_cpu vs cpp_cuda: {((python_cpu - cpp_cuda.cpu())**2).mean().sqrt().item()}")
print(f"RMSE cpp_cpu vs cpp_cuda: {((cpp_cpu - cpp_cuda.cpu())**2).mean().sqrt().item()}")
if enable_python:
    print(f"python cpu: {python_cpu_end_time - python_cpu_start_time}")
    print(f"python cuda: {python_cuda_end_time - python_cuda_start_time}")
print(f"cpp cpu: {cpp_cpu_end_time - cpp_cpu_start_time}")
print(f"cpp cuda: {cpp_cuda_end_time - cpp_cuda_start_time}")

exit(0)

print(f"====== requires_grad = True ======")
mixture.requires_grad = True;
xes.requires_grad = True;
cuda_mixture.requires_grad = True;
cuda_xes.requires_grad = True;
print("warming up")
if enable_python:
    python_cpu = gm.old_evaluate_inversed(mixture, xes)
    python_cpu.sum().backward()
    python_cuda = gm.old_evaluate_inversed(cuda_mixture, cuda_xes)
    python_cuda.sum().backward()
cpp_cpu = gm.evaluate_inversed(mixture, xes)
cpp_cpu.sum().backward()
cpp_cuda = gm.evaluate_inversed(cuda_mixture, cuda_xes)
cpp_cuda.sum().backward()
torch.cuda.synchronize()
mixture.grad = None
xes.grad = None
cuda_mixture.grad = None
cuda_xes.grad = None


if enable_python:
    print("python cpu started")
    python_cpu_start_time = time.perf_counter()
    python_cpu = gm.old_evaluate_inversed(mixture, xes)
    python_cpu_forward_time = time.perf_counter()
    python_cpu.sum().backward()
    python_cpu_end_time = time.perf_counter()
    python_cpu_mixture_grad = mixture.grad.clone()
    python_cpu_xes_grad = xes.grad.clone()
    python_cpu_time = python_cpu_end_time - python_cpu_start_time

    print("python cuda started")
    python_cuda_start_time = time.perf_counter()
    python_cuda = gm.old_evaluate_inversed(cuda_mixture, cuda_xes)
    torch.cuda.synchronize()
    python_cuda_forward_time = time.perf_counter()
    python_cuda.sum().backward()
    torch.cuda.synchronize()
    python_cuda_end_time = time.perf_counter()
    python_cuda_mixture_grad = cuda_mixture.grad.clone()
    python_cuda_xes_grad = cuda_xes.grad.clone()
    python_cuda_time = python_cuda_end_time - python_cuda_start_time

mixture.grad = None
xes.grad = None
cuda_mixture.grad = None
cuda_xes.grad = None

print("cpp cpu started")
cpp_cpu_start_time = time.perf_counter()
cpp_cpu = gm.evaluate_inversed(mixture, xes)
cpp_cpu_forward_time = time.perf_counter()
cpp_cpu.sum().backward()
cpp_cpu_end_time = time.perf_counter()
cpp_cpu_mixture_grad = mixture.grad.clone()
cpp_cpu_xes_grad = xes.grad.clone()
cpp_cpu_time = cpp_cpu_end_time - cpp_cpu_start_time


print("cpp cuda started")
cpp_cuda_start_time = time.perf_counter()
cpp_cuda = gm.evaluate_inversed(cuda_mixture, cuda_xes)
torch.cuda.synchronize()
cpp_cuda_forward_time = time.perf_counter()
cpp_cuda.sum().backward()
torch.cuda.synchronize()
cpp_cuda_end_time = time.perf_counter()
cpp_cuda_mixture_grad = cuda_mixture.grad.clone()
cpp_cuda_xes_grad = cuda_xes.grad.clone()
cpp_cuda_time = cpp_cuda_end_time - cpp_cuda_start_time



if enable_python:
    print(f"RMSE mixture grad python_cpu vs python_cuda: {((python_cpu_mixture_grad - python_cuda_mixture_grad.cpu())**2).mean().sqrt().item()}")
    print(f"RMSE mixture grad python_cpu vs cpp_cpu:     {((python_cpu_mixture_grad - cpp_cpu_mixture_grad)**2).mean().sqrt().item()}")
    print(f"RMSE mixture grad python_cpu vs cpp_cuda:    {((python_cpu_mixture_grad - cpp_cuda_mixture_grad.cpu())**2).mean().sqrt().item()}")
print(f"RMSE mixture grad cpp_cpu vs cpp_cuda:       {((cpp_cpu_mixture_grad - cpp_cuda_mixture_grad.cpu())**2).mean().sqrt().item()}")

if enable_python:
    print(f"RMSE xes grad python_cpu vs python_cuda:     {((python_cpu_xes_grad - python_cuda_xes_grad.cpu())**2).mean().sqrt().item()}")
    print(f"RMSE xes grad python_cpu vs cpp_cpu:         {((python_cpu_xes_grad - cpp_cpu_xes_grad)**2).mean().sqrt().item()}")
    print(f"RMSE xes grad python_cpu vs cpp_cuda:        {((python_cpu_xes_grad - cpp_cuda_xes_grad.cpu())**2).mean().sqrt().item()}")
print(f"RMSE xes grad cpp_cpu vs cpp_cuda:           {((cpp_cpu_xes_grad - cpp_cuda_xes_grad.cpu())**2).mean().sqrt().item()}")

if enable_python:
    print(f"python cpu forward:   {python_cpu_forward_time - python_cpu_start_time}")
    print(f"python cpu backward:  {python_cpu_end_time - python_cpu_forward_time}")
    print(f"python cuda forward:  {python_cuda_forward_time - python_cuda_start_time}")
    print(f"python cuda backward: {python_cuda_end_time - python_cuda_forward_time}")
print(f"cpp cpu forward:      {cpp_cpu_forward_time - cpp_cpu_start_time}")
print(f"cpp cpu backward:     {cpp_cpu_end_time - cpp_cpu_forward_time}")
print(f"cpp cuda forward:     {cpp_cuda_forward_time - cpp_cuda_start_time}")
print(f"cpp cuda backward:    {cpp_cuda_end_time - cpp_cuda_forward_time}")

if enable_python:
    print(f"python min time / cpp min time = {min(python_cpu_time, python_cuda_time) / min(cpp_cpu_time, cpp_cuda_time)}")
    print(f"python cpu time / cpp cpu time = {python_cpu_time / cpp_cpu_time}")
    print(f"python cuda time / cpp cuda time = {python_cuda_time / cpp_cuda_time}")
print(f"cpp cpu time / cpp cuda time = {cpp_cpu_time / cpp_cuda_time}")
