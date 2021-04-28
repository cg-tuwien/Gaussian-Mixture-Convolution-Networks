import torch
import time

device = 'cuda'
dim = 200

start = time.time()

tensor = torch.eye(dim, device=device)

for i in range(0,100):
    mult = torch.rand(dim, device=device)
    tensor = tensor * mult

end = time.time()

print("Execution Time: ", (end - start))

# Dim: 100
# Cpu:  0.010015487670898438
# Cuda: 1.5258831977844238
#
# Dim: 200
# Cpu:  0.0029718875885009766
# Cuda: 1.593775749206543