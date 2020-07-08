import torch
import typing

from torch import Tensor


# from https://discuss.pytorch.org/t/batched-index-select/9115/10
# I added an unit test, but it's easy to make an error in such code, so.. :)
def batched_index_select(input: Tensor, dim: int, index: Tensor) -> Tensor:
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def my_index_select(input: Tensor, index: Tensor) -> Tensor:
    dim = len(index.shape)-1
    expanse = list(input.shape)
    for i in range(len(index.shape)):
        expanse[i] = -1
    for ii in range(len(index.shape), len(input.shape)):
        index = index.unsqueeze(ii)
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def flatten_index(input: Tensor, shape: torch.Size) -> Tensor:
    assert input.shape[-1] == len(shape)
    shape = list(shape)
    old_m = 1
    for d in range(1, len(shape)+1):
        new_m = old_m * shape[-d]
        shape[-d] = old_m
        old_m = new_m
    shape = torch.tensor(shape, dtype=torch.long, device=input.device)

    # dot product
    return (input * shape).sum(dim=-1).view(input.shape[:-1])


def gen_random_positive_definite(dims: typing.List, epsilon: float = 0.01, device: torch.device = 'cpu') -> Tensor:
    assert dims[-1] == dims[-2]
    A = torch.rand(dims, dtype=torch.float32, device=device) * 2 - 1
    return A @ A.transpose(-1, -2) + torch.eye(dims[-1], dtype=torch.float32, device=device) * epsilon


def batched_trace(matrices: Tensor) -> Tensor:
    return torch.diagonal(matrices, dim1=-2, dim2=-1).sum(-1)
