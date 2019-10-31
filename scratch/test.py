import torch
import numpy as np

#torch.
A = np.array([[1.]]).transpose()
M = torch.tensor(A)


#M = np.array([[1.]]).transpose()
#M = torch.tensor(M)
# M. gives all, any, argmax.. completions if the next lines are commented

#M = np.array([[1.]])
#M = torch.tensor(M)
# M. gives partition, all, any, argmax..

#M = torch.tensor(np.array([[1.]]))
# M. gives nothing if the previous lines are commented

#M = torch.Tensor([[1.]])
# M. gives nothing if the previous lines are commented
