import math

import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt

import mat_tools

class Mixture:
    def __init__(self, factors: Tensor, positions: Tensor, covariances: Tensor) -> None:
        assert len(factors.size()) == 1
        nComponents = factors.size()[0]
        self.dimensions = positions.size()[0]
        assert nComponents == positions.size()[1]
        assert nComponents == covariances.size()[1]
        assert torch.all(mat_tools.triangle_det(covariances) > 0)
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
        selected = torch.zeros(xes.size()[1], dtype=torch.long)
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
    covs = mat_tools.gen_random_positive_definite_triangle(n, dims) * cov_radius

    return Mixture(factors, positions, covs)


# todo: this function is a mess
def generate_null_mixture(n: int, dims: int) -> Mixture:
    assert dims == 2 or dims == 3
    assert n > 0

    factors = torch.zeros(n, dtype=torch.float)
    positions = torch.zeros(dims, n, dtype=torch.float)
    covs = mat_tools.gen_identity_triangle(n, dims)
    m = Mixture(factors, positions, covs)
    m.covariances = mat_tools.gen_null_triangle(n, dims)
    return m


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
    detc1tc2 = mat_tools.triangle_det(m1_c) * mat_tools.triangle_det(m2_c)
    detc1pc2 = mat_tools.triangle_det(covariances)
    factors = math.pow(math.sqrt(2 * math.pi), m1.dimensions) * m1_f * m2_f * torch.sqrt(detc1tc2) / torch.sqrt(detc1pc2)
    return Mixture(factors, positions, covariances)
