import torch

from torch import Tensor


def gen_random_positive_definite_triangle(n: int, dims: int) -> Tensor:
    assert dims == 2 or dims == 3
    retval = torch.zeros(3 if dims == 2 else 6, n)
    for i in range(n):
        A = torch.rand(dims, dims) * 2 - 1
        A = A @ A.t() + torch.eye(dims) * 0.01
        if dims == 2:
            retval[0:2, i] = A[0, :]
            retval[2, i] = A[1, 1]
        else:
            retval[0:3, i] = A[0, :]
            retval[3:5, i] = A[1, 1:]
            retval[5, i] = A[2, 2]
    return retval


def gen_null_triangle(n: int, dims: int) -> Tensor:
    assert dims == 2 or dims == 3
    return torch.zeros(3 if dims == 2 else 6, n, dtype=torch.float)


def gen_identity_triangle(n: int, dims: int) -> Tensor:
    assert dims == 2 or dims == 3
    covs = torch.zeros(3 if dims == 2 else 6, n, dtype=torch.float)
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
                             a[1] * a[1]])
    else:
        return torch.tensor([a[0] * a[0],
                             a[0] * a[1],
                             a[0] * a[2],
                             a[1] * a[1],
                             a[1] * a[2],
                             a[2] * a[2]]);


def triangle_xAx(A: Tensor, xes: Tensor) -> Tensor:
    assert (xes.size()[0] == 2 and A.size()[0] == 3) or (xes.size()[0] == 3 and A.size()[0] == 6)
    if xes.size()[0] == 2:
        return A[0] * xes[0] * xes[0] + 2 * A[1] * xes[0] * xes[1] + A[2] * xes[1] * xes[1]
    return (A[0] * xes[0] * xes[0]
            + 2 * A[1] * xes[0] * xes[1]
            + 2 * A[2] * xes[0] * xes[2]
            + A[3] * xes[1] * xes[1]
            + 2 * A[4] * xes[1] * xes[2]
            + A[5] * xes[2] * xes[2])


def triangle_det(A: Tensor) -> Tensor:
    n_triangle_elements = A.size()[0]
    assert n_triangle_elements == 3 or n_triangle_elements == 6
    if n_triangle_elements == 3:
        return A[0] * A[2] - A[1] * A[1]
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

        result[2] = A[1] * B[1] + A[3] * B[3] + A[4] * B[4]
        result[2] = A[1] * B[2] + A[3] * B[4] + A[4] * B[5]

        result[2] = A[2] * B[2] + A[4] * B[4] + A[5] * B[5]
    return result
