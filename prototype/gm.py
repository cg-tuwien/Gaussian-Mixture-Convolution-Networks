import math

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
        assert torch.all(_triangle_determinants(covariances) > 0)
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
    
    def evaluate_component(self, xes: Tensor, component: int) -> Tensor:
        v = xes - self.positions[:, component].view(-1, 1).expand_as(xes)
        cov_tri = self.covariances[:, component]
        cov = self._mat_from_tri(cov_tri)
        cov_i = cov.cholesky().cholesky_inverse()
        # v.t() @ cov_i @ v, but with more than one column vector
        v = -0.5 * (v * (cov_i @ v)).sum(dim=0)
        v = self.factors[component] * torch.exp(v)
        return v
    
    def evaluate(self, xes: Tensor) -> Tensor:
        values = torch.zeros(xes.size()[1])
        for i in range(self.number_of_components()):
            values += self.evaluate_component(xes, i)
        return values
    
    def max_component(self, xes: Tensor) -> Tensor:
        selected = torch.zeros(xes.size()[1], dtype=torch.int)
        values = self.evaluate_component(xes, 0)
        for i in range(self.number_of_components()):
            component_values = self.evaluate_component(xes, i)
            mask = component_values > values
            selected[mask] = i
            values[mask] = component_values[mask]
        
        return selected
            
    def debug_show(self, x_low: float = -22, y_low: float = -22, x_high: float = 22, y_high: float = 22, step: float = 0.1) -> Tensor:
        xv, yv = torch.meshgrid([torch.arange(x_low, x_high, step, dtype=torch.float), torch.arange(y_low, y_high, step, dtype=torch.float)])
        xes = torch.cat((xv.reshape(1, -1), yv.reshape(1, -1)), 0)
        values = self.evaluate(xes)
        image = values.view(xv.size()[0], xv.size()[1]).numpy()
        plt.imshow(image)
        plt.show()
        return image

def _gen_random_covs(n: int, dims: int) -> Tensor:
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

def _gen_null_covs(n: int, dims: int) -> Tensor:
    assert dims == 2 or dims == 3
    return torch.zeros(3 if dims == 2 else 6, n)

def _gen_identity_covs(n: int, dims: int) -> Tensor:
    assert dims == 2 or dims == 3
    covs = torch.zeros(3 if dims == 2 else 6, n)
    if dims == 2:
        covs[0, :] = 1
        covs[2, :] = 1
    else:
        covs[0, :] = 1
        covs[3, :] = 1
        covs[5, :] = 1
    return covs

# we will need to work on the initialisation. it's unlikely this simple one will work.
def generate_random_mixtures(n: int, dims: int,
                             pos_radius: float = 10,
                             cov_radius: float = 10,
                             factor_min: float = -1,
                             factor_max: float = 1) -> Mixture:
    assert dims == 2 or dims == 3
    assert factor_min < factor_max
    assert n > 0

    factors = torch.rand(n) * (factor_max - factor_min) + factor_min
    positions = torch.rand(dims, n) * 2 * pos_radius - pos_radius
    covs = _gen_random_covs(n, dims) * cov_radius

    return Mixture(factors, positions, covs)

# todo: this function is a mess
def generate_null_mixture(n: int, dims: int) -> Mixture:
    assert dims == 2 or dims == 3
    assert n > 0

    factors = torch.zeros(n)
    positions = torch.zeros(dims, n)
    covs = _gen_identity_covs(n, dims)
    m = Mixture(factors, positions, covs)
    m.covariances = _gen_null_covs(n, dims)
    return m

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


def _triangle_determinants(A: Tensor) -> Tensor:
    n_triangle_elements = A.size()[0]
    assert n_triangle_elements == 3 or n_triangle_elements == 6
    if n_triangle_elements == 3:
        return A[0] * A[2] - A[1] * A[1]
    return A[0]*A[3]*A[5] + 2 * A[1] * A[4] * A[2] - A[2] * A[2] * A[3] - A[1] * A[1] * A[5] - A[0] * A[4] * A[4]

def _triangle_matmul(A: Tensor, B: Tensor) -> Tensor:
    n_triangle_elements = A.size()[0]
    assert n_triangle_elements == 3 or n_triangle_elements == 6
    assert  A.size() == B.size()
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

def _polynomMulRepeat(A: Tensor, B: Tensor) -> (Tensor, Tensor):
    if len(A.size()) == 2:
        assert A.size()[0] == B.size()[0]
        A_n = A.size()[1]
        B_n = B.size()[1]
        return (A.repeat(1, B_n), B.repeat_interleave(A_n, 1))
    else:
        A_n = A.size()[0]
        B_n = B.size()[0]
        return (A.repeat(B_n), B.repeat_interleave(A_n))

def convolve(m1: Mixture, m2: Mixture) -> Mixture:
    assert m1.dimensions == m2.dimensions
    m1_f, m2_f = _polynomMulRepeat(m1.factors, m2.factors)
    m1_p, m2_p = _polynomMulRepeat(m1.positions, m2.positions)
    m1_c, m2_c = _polynomMulRepeat(m1.covariances, m2.covariances)

    positions = m1_p + m2_p
    covariances = m1_c + m2_c
    detc1tc2 = _triangle_determinants(m1_c) * _triangle_determinants(m2_c)
    detc1pc2 = _triangle_determinants(covariances)
    factors = math.pow(math.sqrt(2 * math.pi), m1.dimensions) * m1_f * m2_f * torch.sqrt(detc1tc2) / torch.sqrt(detc1pc2)
    return Mixture(factors, positions, covariances)
