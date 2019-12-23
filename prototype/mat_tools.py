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


def trimat_size(dims: int) -> int:
    return 3 if dims == 2 else 6


def gen_random_positive_definite(dims: typing.List, epsilon: float = 0.01, device: torch.device = 'cpu') -> Tensor:
    assert dims[-1] == dims[-2]
    A = torch.rand(dims, dtype=torch.float32, device=device) * 2 - 1
    return A @ A.transpose(-1, -2) + torch.eye(dims[-1], dtype=torch.float32, device=device) * epsilon


def gen_random_positive_definite_triangle(n: int, dims: int, device: torch.device = 'cpu') -> Tensor:
    assert dims == 2 or dims == 3
    retval = torch.rand(n, dims, dims, dtype=torch.float32, device=device) * 2 - 1
    retval = retval.transpose(1, 2) @ retval + torch.eye(dims, dtype=torch.float32, device=device) * 0.01
    return normal_to_triangle(retval.transpose(0, 2))


def gen_null_triangle(n: int, dims: int, device: torch.device = 'cpu') -> Tensor:
    assert dims == 2 or dims == 3
    return torch.zeros(3 if dims == 2 else 6, n, dtype=torch.float, device=device)


def gen_identity_triangle(n: int, dims: int, device: torch.device = 'cpu') -> Tensor:
    assert dims == 2 or dims == 3
    covs = torch.zeros(trimat_size(dims), n, dtype=torch.float, device=device)
    if dims == 2:
        covs[0, :] = 1
        covs[2, :] = 1
    else:
        covs[0, :] = 1
        covs[3, :] = 1
        covs[5, :] = 1
    return covs


def triangle_outer_product(a: Tensor) -> Tensor:
    if a.numel() == 2:
        return torch.tensor([a[0] * a[0],
                             a[0] * a[1],
                             a[1] * a[1]], dtype=torch.float32, device=a.device)
    else:
        return torch.tensor([a[0] * a[0],
                             a[0] * a[1],
                             a[0] * a[2],
                             a[1] * a[1],
                             a[1] * a[2],
                             a[2] * a[2]], dtype=torch.float32, device=a.device);


def triangle_xAx(A: Tensor, xes: Tensor) -> Tensor:
    assert (xes.size()[0] == 2 and A.size()[0] == 3) or (xes.size()[0] == 3 and A.size()[0] == 6)
    if xes.size()[0] == 2:
        xes_squared = xes ** 2
        xes_cross = xes[0] * xes[1]
        return A[0] * xes_squared[0] + (2 * A[1]) * xes_cross + A[2] * xes_squared[1]
    return (A[0] * xes[0] * xes[0]
            + 2 * A[1] * xes[0] * xes[1]
            + 2 * A[2] * xes[0] * xes[2]
            + A[3] * xes[1] * xes[1]
            + 2 * A[4] * xes[1] * xes[2]
            + A[5] * xes[2] * xes[2])


def triangle_det(A: Tensor) -> Tensor:
    n_triangle_elements = A.size()[0]
    # assert n_triangle_elements == 3 or n_triangle_elements == 6
    if n_triangle_elements == 3:
        return A[0] * A[2] - A[1] ** 2
    return A[0] * A[3] * A[5] + 2 * A[1] * A[4] * A[2] - A[2] * A[2] * A[3] - A[1] * A[1] * A[5] - A[0] * A[4] * A[
        4]


def triangle_matmul(A: Tensor, B: Tensor) -> Tensor:
    n_triangle_elements = A.size()[0]
    assert n_triangle_elements == 3 or n_triangle_elements == 6
    assert A.size() == B.size()
    result = torch.empty(n_triangle_elements, A.size()[1])
    if n_triangle_elements == 3:
        result[0] = A[0] * B[0] + A[1] * B[1]
        result[1] = A[0] * B[1] + A[1] * B[2]
        result[2] = A[1] * B[1] + A[2] * B[2]
    else:
        result[0] = A[0] * B[0] + A[1] * B[1] + A[2] * B[2]
        result[1] = A[0] * B[1] + A[1] * B[3] + A[2] * B[4]
        result[2] = A[0] * B[2] + A[1] * B[4] + A[2] * B[5]

        result[3] = A[1] * B[1] + A[3] * B[3] + A[4] * B[4]
        result[4] = A[1] * B[2] + A[3] * B[4] + A[4] * B[5]

        result[5] = A[2] * B[2] + A[4] * B[4] + A[5] * B[5]
    return result


def triangle_invert(tris: Tensor) -> Tensor:
    mats = triangle_to_normal(tris)
    mats = mats.transpose(0, 2).inverse().transpose(0, 2)
    ## cholesky would be quicker, but batchwise cholesky_inverse is not implemented as of pytorch 1.3
    # mats = mats.transpose(0, 2).cholesky().cholesky_inverse().transpose(0, 2)

    return normal_to_triangle(mats)


# takes triangle_vals x n, returns rows x columns x n
def triangle_to_normal(A: Tensor) -> Tensor:
    a_single_one = False
    if len(A.size()) == 1:
        a_single_one = True
        A = A.view(-1, 1)
    dims = int(A.size()[0] // 3 + 1)
    O = torch.empty(dims, dims, A.size()[1], device=A.device)
    if dims == 2:
        O[0, 0, :] = A[0, :]
        O[0, 1, :] = A[1, :]
        O[1, 0, :] = A[1, :]
        O[1, 1, :] = A[2, :]
    elif dims == 3:
        O[0, 0, :] = A[0, :]
        O[0, 1, :] = A[1, :]
        O[0, 2, :] = A[2, :]
        O[1, 0, :] = A[1, :]
        O[1, 1, :] = A[3, :]
        O[1, 2, :] = A[4, :]
        O[2, 0, :] = A[2, :]
        O[2, 1, :] = A[4, :]
        O[2, 2, :] = A[5, :]
    else:
        assert False
    if a_single_one:
        return O.view(dims, dims)
    else:
        return O

# takes rows x columns x n, returns triangle_vals x n
def normal_to_triangle(A: Tensor) -> Tensor:
    dims = A.size()[0]
    assert A.size()[1] == dims

    a_single_one = False
    if len(A.size()) == 2:
        a_single_one = True
        A = A.view(dims, dims, 1)

    O = torch.empty(int((dims-1) * 3), A.size()[2], device=A.device)
    if dims == 2:
        O[0, :] = A[0, 0, :]
        O[1, :] = A[0, 1, :]
        O[2, :] = A[1, 1, :]
    elif dims == 3:
        O[0, :] = A[0, 0, :]
        O[1, :] = A[0, 1, :]
        O[2, :] = A[0, 2, :]
        O[3, :] = A[1, 1, :]
        O[4, :] = A[1, 2, :]
        O[5, :] = A[2, 2, :]
    else:
        assert False
    if a_single_one:
        return O[:]
    else:
        return O

