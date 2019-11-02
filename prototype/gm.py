import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt


class Mixture:
    def __init__(self, factors: Tensor, positions: Tensor, covariances: Tensor) -> None:
        assert len(factors.size()) == 1
        nComponents = factors.size()[0]
        self.dimensions = positions.size()[0]
        assert nComponents == positions.size()[1]
        assert nComponents == covariances.size()[1]
        assert torch.all(_determinants(covariances) > 0)
        if self.dimensions == 2:
            # upper triangle of a 2x2 matrix has 3 elements
            assert covariances.size()[0] == 3
        else:
            # upper triangle of a 3x3 matrix has 6 elements
            assert covariances.size()[0] == 6
        
        self.factors = factors
        self.positions = positions
        self.covariances = covariances

    def number_of_components(self):
        return self.factors.size()[0]

    def _mat_from_tri(self, m: Tensor) -> Tensor:
        if m.size()[0] == 3:
            return torch.tensor([[m[0], m[1]],
                                 [m[1], m[2]]])
        return torch.tensor([[m[0], m[1], m[2]],
                             [m[1], m[3], m[4]],
                             [m[2], m[4], m[5]]])
    
    def evaluate(self, xes: Tensor) -> Tensor:
        values = torch.zeros(xes.size()[1])
        for i in range(self.number_of_components()):
            v = xes - self.positions[:, i].view(-1, 1).expand_as(xes)
            cov_tri = self.covariances[:, i]
            cov = self._mat_from_tri(cov_tri)
            cov_i = cov.cholesky().cholesky_inverse()
            # v.t() @ cov_i @ v, but with more than one column vector
            v = -0.5 * (v * (cov_i @ v)).sum(dim=0)
            values += self.factors[i] * torch.exp(v)
        return values
            
    def debug_show(self, width: int, height: int) -> Tensor:
        xv, yv = torch.meshgrid([torch.arange(-22, 22, 0.1), torch.arange(-22, 22, 0.1)])
        xes = torch.cat((xv.reshape(1, -1), yv.reshape(1, -1)), 0)
        values = self.evaluate(xes)
        image = values.view(xv.size()[0], xv.size()[1]).numpy()
        plt.imshow(image)
        plt.show()
        return image

#todo seems like the covs are corellated. when plotting, they are all tl-br and few or none are tr-bl
def _gen_random_covs(n: int, dims: int) -> Tensor:
    assert dims == 2 or dims == 3
    retval = torch.zeros(3 if dims == 2 else 6, n)
    for i in range(n):
        A = torch.rand(dims, dims) * 4
        A = A @ A.t() + torch.eye(dims) * 0.04
        if dims == 2:
            retval[0:2, i] = A[0, :]
            retval[2, i] = A[1, 1]
        else:
            retval[0:3, i] = A[0, :]
            retval[3:5, i] = A[1, 1:]
            retval[5, i] = A[2, 2]
    return retval

#todo: will need more work on seeding / distribution
def generate_random_mixtures(n: int, dims: int) -> Mixture:
    assert dims == 2 or dims == 3
    assert n > 0

    factors = torch.rand(n) * 2 - 1
    positions = torch.rand(dims, n) * 20 - 10
    covs = _gen_random_covs(n, dims)
    return Mixture(factors, positions, covs)

def _xAx_withTriangleA(A: Tensor, xes: Tensor) -> Tensor:
    assert (xes.size()[0] == 2 and A.size()[0] == 3) or (xes.size()[0] == 3 and A.size()[0] == 6)
    if xes.size()[0] == 2:
        return A[0] * xes[0] * xes[0] + 2 * A[1] * xes[0] * xes[1] + A[2] * xes[1] * xes[1]
    return (A[0] * xes[0] * xes[0]
            + 2 * A[1] * xes[0] * xes[1]
            + 2 * A[2] * xes[0] * xes[2]
            + A[3] * xes[1] * xes[1]
            + 2 * A[4] * xes[1] * xes[2]
            + A[5] * xes[2] * xes[2])


def _determinants(A: Tensor) -> Tensor:
    n_triangle_elements = A.size()[0]
    assert n_triangle_elements == 3 or n_triangle_elements == 6
    if n_triangle_elements == 3:
        return A[0] * A[2] - A[1] * A[1]
    return A[0]*A[3]*A[5] + 2 * A[1] * A[4] * A[2] - A[2] * A[2] * A[3] - A[1] * A[1] * A[5] - A[0] * A[4] * A[4]


def _polynomMulRepeat(A: Tensor, B: Tensor) -> (Tensor, Tensor):
    assert A.size()[0] == B.size()[0]
    A_n = A.size()[1]
    B_n = B.size()[1]
    return (A.repeat(1, B_n), B.repeat_interleave(A_n, 1))


def convolve(m1: Mixture, m2: Mixture) -> Mixture:
    assert m1.dimensions == m2.dimensions
    nComponents = m1.number_of_components() * m2.number_of_components()
    
    
