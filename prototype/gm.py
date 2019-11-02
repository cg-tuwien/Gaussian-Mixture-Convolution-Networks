import torch
from torch import Tensor
import matplotlib.pyplot as plt


class Mixture:
    def __init__(self, factors: Tensor, positions: Tensor, covariances: Tensor) -> None:
        assert len(factors.size()) == 1
        nComponents = factors.size()[0]
        self.dimensions = positions.size()[0]
        assert nComponents == positions.size()[1]
        assert nComponents == covariances.size()[1]
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
            
    def debug_show(self, width: int, height: int) -> None:
        ...


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
    
    
