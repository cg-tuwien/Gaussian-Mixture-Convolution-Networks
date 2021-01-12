import torch
import typing

from torch import Tensor

from .cpp.extensions.math import matrix_inverse as cppExtensionsMathMatrixInverse


# from https://discuss.pytorch.org/t/batched-index-select/9115/10
# I added an unit test, but it's easy to make an error in such code, so.. :)
def batched_index_select(tensor: Tensor, dim: int, index: Tensor) -> Tensor:
    for ii in range(1, len(tensor.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(tensor.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(tensor, dim, index)


def my_index_select(tensor: Tensor, index: Tensor) -> Tensor:
    dim = len(index.shape)-1
    expanse = list(tensor.shape)
    for i in range(len(index.shape)):
        expanse[i] = -1 if index.shape[i] > 1 else tensor.shape[i]
    for ii in range(len(index.shape), len(tensor.shape)):
        index = index.unsqueeze(ii)
    index = index.expand(expanse)
    return torch.gather(tensor, dim, index)


def inverse(tensor: Tensor) -> Tensor:
    """faster version of inverse (compared to pytorch) for 2x2 and 3x3 matrices"""
    return cppExtensionsMathMatrixInverse.apply(tensor)  # .transpose(-1, -2)


def flatten_index(tensor: Tensor, shape: torch.Size) -> Tensor:
    assert tensor.shape[-1] == len(shape)
    shape = list(shape)
    old_m = 1
    for d in range(1, len(shape)+1):
        new_m = old_m * shape[-d]
        shape[-d] = old_m
        old_m = new_m
    shape = torch.tensor(shape, dtype=torch.long, device=tensor.device)

    # dot product
    return (tensor * shape).sum(dim=-1).view(tensor.shape[:-1])


def gen_random_positive_definite(dims: typing.List, epsilon: float = 0.01, device: torch.device = 'cpu') -> Tensor:
    assert dims[-1] == dims[-2]
    A = torch.rand(dims, dtype=torch.float32, device=device) * 2 - 1
    return A @ A.transpose(-1, -2) + torch.eye(dims[-1], dtype=torch.float32, device=device) * epsilon


def batched_trace(matrices: Tensor) -> Tensor:
    return torch.diagonal(matrices, dim1=-2, dim2=-1).sum(-1)
