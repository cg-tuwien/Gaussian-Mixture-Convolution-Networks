import math

import torch
from pygments.lexer import include
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt

import mat_tools


class Mixture:
    # maybe it's faster if we put everything into one matrix (data more local).
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
        self.inverted_covariances = mat_tools.triangle_invert(covariances)

    def device(self):
        return self.factors.device

    def number_of_components(self):
        return self.factors.size()[0]

    def evaluate_component_many_xes(self, xes: Tensor, component: int) -> Tensor:
        factors = self.factors[component]
        position = self.positions[:, component].view(-1, 1)
        cov_i_trimat = self.inverted_covariances[:, component]
        v = xes - position.expand_as(xes)
        # cov_i = mat_tools.triangle_to_normal(cov_i_trimat)

        ## v.t() @ cov_i @ v, but with more than one column vector
        # v = -0.5 * (v * (cov_i @ v)).sum(dim=0)
        v = -0.5 * mat_tools.triangle_xAx(cov_i_trimat, v)

        v = factors * torch.exp(v)
        assert not torch.isnan(v).any()
        assert not torch.isinf(v).any()
        return v

    def evaluate_few_xes_component_wise(self, xes: Tensor) -> Tensor:
        n_xes = xes.size()[1]
        # xes in cols, comps in rows
        values = torch.zeros(self.number_of_components(), n_xes, dtype=torch.float32)
        for i in range(n_xes):
            vs = xes[:, i].view(-1, 1).expand_as(self.positions) - self.positions
            vs = -0.5 * mat_tools.triangle_xAx(self.inverted_covariances, vs)
            vs = self.factors * torch.exp(vs)
            values[:, i] = vs
        return values

    def evaluate_few_xes(self, xes: Tensor) -> Tensor:
        return self.evaluate_few_xes_component_wise(xes).sum(0)

    def evaluate_many_xes(self, xes: Tensor) -> Tensor:
        assert (mat_tools.triangle_det(self.inverted_covariances) > 0).all()
        values = torch.zeros(xes.size()[1], dtype=torch.float32, device=xes.device)
        for i in range(self.number_of_components()):
            # todo: adding many components like this probably makes the gradient graph and therefore memory explode
            values += self.evaluate_component_many_xes(xes, i)
        return values
    
    def max_component_many_xes(self, xes: Tensor) -> Tensor:
        selected = torch.zeros(xes.size()[1], dtype=torch.long)
        values = self.evaluate_component_many_xes(xes, 0)
        for i in range(self.number_of_components()):
            component_values = self.evaluate_component_many_xes(xes, i)
            mask = component_values > values
            selected[mask] = i
            values[mask] = component_values[mask]
        
        return selected
            
    def debug_show(self, x_low: float = -22, y_low: float = -22, x_high: float = 22, y_high: float = 22, step: float = 0.1) -> Tensor:
        xv, yv = torch.meshgrid([torch.arange(x_low, x_high, step, dtype=torch.float, device=self.factors.device),
                                 torch.arange(y_low, y_high, step, dtype=torch.float, device=self.factors.device)])
        xes = torch.cat((xv.reshape(1, -1), yv.reshape(1, -1)), 0)
        values = self.evaluate_many_xes(xes).detach()
        image = values.view(xv.size()[0], xv.size()[1]).cpu().numpy()
        plt.imshow(image)
        plt.show()
        return image

    def cuda(self):
        return Mixture(self.factors.cuda(), self.positions.cuda(), self.covariances.cuda())

    def cpu(self):
        return Mixture(self.factors.cpu(), self.positions.cpu(), self.covariances.cpu())

    def detach(self):
        detached_mixture = generate_null_mixture(1, self.dimensions, device=self.device())
        detached_mixture.factors = self.factors.detach()
        detached_mixture.positions = self.positions.detach()
        detached_mixture.covariances = self.covariances.detach()
        detached_mixture.inverted_covariances = self.inverted_covariances.detach()
        return detached_mixture

    def save(self, file_name: str):
        dict = {
            "type": "gm.Mixture",
            "version": 1,
            "weights": self.factors,
            "positions": self.positions,
            "covariances": self.covariances
        }
        torch.save(dict, "/home/madam/temp/prototype/" + file_name)

    @classmethod
    def load(cls, file_name: str):
        dict = torch.load("/home/madam/temp/prototype/" + file_name)
        assert dict["type"] == "gm.Mixture"
        assert dict["version"] == 1
        return Mixture(dict["weights"], dict["positions"], dict["covariances"])


class ConvolutionLayer:
    def __init__(self, mixture: Mixture, bias: Tensor):
        assert bias > 0
        self.mixture = mixture
        self.bias = bias

    def evaluate_few_xes(self, positions: Tensor):
        values = self.mixture.evaluate_few_xes(positions) - self.bias
        return torch.max(values, torch.tensor([0.0001], dtype=torch.float32, device=self.mixture.device()))

    def debug_show(self, x_low: float = -22, y_low: float = -22, x_high: float = 22, y_high: float = 22, step: float = 0.1) -> Tensor:
        m = self.mixture.detach()
        xv, yv = torch.meshgrid([torch.arange(x_low, x_high, step, dtype=torch.float, device=m.device()),
                                 torch.arange(y_low, y_high, step, dtype=torch.float, device=m.device())])
        xes = torch.cat((xv.reshape(1, -1), yv.reshape(1, -1)), 0)
        values = m.evaluate_many_xes(xes)
        values -= self.bias.detach()
        values[values < 0] = 0
        image = values.view(xv.size()[0], xv.size()[1]).cpu().numpy()
        plt.imshow(image)
        plt.show()
        return image

    def device(self):
        return self.mixture.device()

# we will need to work on the initialisation. it's unlikely this simple one will work.
def generate_random_mixtures(n: int, dims: int,
                             pos_radius: float = 10,
                             cov_radius: float = 10,
                             factor_min: float = -1,
                             factor_max: float = 1,
                             device: torch.device = 'cpu') -> Mixture:
    assert dims == 2 or dims == 3
    assert factor_min < factor_max
    assert n > 0

    factors = torch.rand(n, dtype=torch.float32, device=device) * (factor_max - factor_min) + factor_min
    positions = torch.rand(dims, n, dtype=torch.float32, device=device) * 2 * pos_radius - pos_radius
    covs = mat_tools.gen_random_positive_definite_triangle(n, dims, device=device) * cov_radius

    return Mixture(factors, positions, covs)


# todo: this function is a mess
def generate_null_mixture(n: int, dims: int, device: torch.device = 'cpu') -> Mixture:
    assert dims == 2 or dims == 3
    assert n > 0

    factors = torch.zeros(n, dtype=torch.float, device=device)
    positions = torch.zeros(dims, n, dtype=torch.float, device=device)
    covs = mat_tools.gen_identity_triangle(n, dims, device=device)
    m = Mixture(factors, positions, covs)
    m.covariances = mat_tools.gen_null_triangle(n, dims, device=device)
    return m


def _polynomMulRepeat(A: Tensor, B: Tensor) -> (Tensor, Tensor):
    if len(A.size()) == 2:
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
