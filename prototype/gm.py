from __future__ import annotations
import math
import typing

import torch
from pygments.lexer import include
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt

import mat_tools


class Mixture:
    # data: first dimension: batch, second dimension: component, third and fourth(?) dimension: data
    def __init__(self, weights: Tensor, positions: Tensor, covariances: Tensor) -> None:
        assert len(weights.size()) == 2
        assert len(positions.size()) == 3
        assert len(covariances.size()) == 4

        self.weights = weights
        self.positions = positions
        self.covariances = covariances
        self._inverted_covariances = None

        assert self.n_components() == weights.size()[1]
        assert self.n_components() == positions.size()[1]
        assert self.n_components() == covariances.size()[1]
        assert covariances.size()[2] == self.n_dimensions()
        assert covariances.size()[3] == self.n_dimensions()
        assert torch.all(covariances.det() > 0)

    def inverted_covariances(self):
        if self._inverted_covariances is None:
            self.update_inverted_covariance()
        return self._inverted_covariances

    def update_inverted_covariance(self) -> None:
        self._inverted_covariances = self.covariances.inverse()

    def update_covariance_from_inverted(self) -> None:
        self.covariances = self._inverted_covariances.inverse()

    def device(self) -> torch.device:
        return self.weights.device

    def n_components(self) -> int:
        return self.weights.size()[1]

    def n_layers(self) -> int:
        return self.weights.size()[0]

    def n_dimensions(self) -> int:
        return self.positions.size()[2]

    def evaluate_few_xes_component_wise(self, xes: Tensor) -> Tensor:
        n_layers = self.n_layers()
        n_dims = self.n_dimensions()
        n_comps = self.n_components()
        n_xes = xes.size()[1]
        assert xes.size()[0] == n_layers
        # xes: first dim: list, second dim; x/y

        if n_layers * n_comps * n_xes < 100 * 1024 * 1024:
            # 1. dim: batches (from mixture), 2. component, 3. xes, 4.+: vector / matrix components
            xes = xes.view(n_layers, 1, -1, n_dims)
            positions = self.positions.view(n_layers, n_comps, 1, n_dims)
            values = xes - positions

            # x^t A x -> quadratic form
            x_t = values.view(n_layers, n_comps, -1, 1, n_dims)
            x = values.view(n_layers, n_comps, -1, n_dims, 1)
            A = self.inverted_covariances().view(n_layers, n_comps, 1, n_dims, n_dims)
            values = -0.5 * x_t @ A @ x
            values = values.view(n_layers, n_comps, -1)
        else:
            # todo: select min of n_layers and n_components or something?
            # todo: test
            batched_values = torch.zeros(n_layers, n_comps, n_xes, dtype=torch.float32, device=self.device())
            for i in range(n_layers):
                xes_slice = xes[i, :, :].view(1, -1, n_dims)
                positions = self.positions[i, :, :].view(n_comps, 1, n_dims)
                values = xes_slice - positions

                # x^t A x -> quadratic form
                x_t = values.view(n_comps, -1, 1, n_dims)
                x = values.view(n_comps, -1, n_dims, 1)
                A = self.inverted_covariances()[i, :, :, :].view(n_comps, 1, n_dims, n_dims)
                values = -0.5 * x_t @ A @ x
                batched_values[i, :, :] = values.view(n_comps, -1)
            values = batched_values

        values = self.weights.view(n_layers, n_comps, 1) * torch.exp(values)
        return values.view(n_layers, n_comps, -1)

    def evaluate_few_xes(self, xes: Tensor) -> Tensor:
        return self.evaluate_few_xes_component_wise(xes).sum(1)

    def evaluate_component_many_xes(self, xes: Tensor, component: int) -> Tensor:
        n_layers = self.n_layers()
        n_dims = self.n_dimensions()
        n_xes = xes.size()[1]
        assert xes.size()[0] == n_layers
        assert xes.size()[2] == n_dims
        assert component < self.n_components()

        weights = self.weights[:, component].view(-1, 1)
        positions = self.positions[:, component, :]
        inverted_covs = self.inverted_covariances()[:, component, :, :]

        # first dimension: batch (from mixture), second: sampling (xes), third: data
        v = xes - positions.view(n_layers, 1, n_dims)
        # first dimension: batch (from mixture), second: sampling (xes), third and fourth: matrix data
        inverted_covs = inverted_covs.view(n_layers, 1, n_dims, n_dims)

        v = -0.5 * v.view(n_layers, n_xes, 1, n_dims) @ inverted_covs @ v.view(n_layers, n_xes, n_dims, 1)
        v = v.view(n_layers, n_xes)
        v = weights * torch.exp(v)
        assert not torch.isnan(v).any()
        assert not torch.isinf(v).any()
        return v

    def evaluate_many_xes(self, xes: Tensor) -> Tensor:
        n_xes = xes.size()[1]
        n_layers = self.n_layers()
        assert n_layers == xes.size()[0]
        assert self.n_dimensions() == xes.size()[2]
        values = torch.zeros(n_layers, n_xes, dtype=torch.float32, device=xes.device)
        for i in range(self.n_components()):
            # todo: adding many components like this probably makes the gradient graph and therefore memory explode
            values += self.evaluate_component_many_xes(xes, i)
        return values

    def max_component_many_xes(self, xes: Tensor) -> Tensor:
        assert self.n_layers() == 1
        selected = torch.zeros(xes.size()[1], dtype=torch.long)
        values = self.evaluate_component_many_xes(xes, 0).view(-1)
        for i in range(self.n_components()):
            component_values = self.evaluate_component_many_xes(xes, i).view(-1)
            mask = component_values > values
            selected[mask] = i
            values[mask] = component_values[mask]

        return selected

    def debug_show(self, layer_i: int = 0, x_low: float = -22, y_low: float = -22, x_high: float = 22, y_high: float = 22, step: float = 0.1, imshow=True) -> Tensor:
        assert layer_i < self.n_layers()
        assert self.n_dimensions() == 2
        m = self.detach().batch(layer_i)

        xv, yv = torch.meshgrid([torch.arange(x_low, x_high, step, dtype=torch.float, device=self.device()),
                                 torch.arange(y_low, y_high, step, dtype=torch.float, device=self.device())])
        xes = torch.cat((xv.reshape(-1, 1), yv.reshape(-1, 1)), 1).view(1, -1, 2)

        values = m.evaluate_many_xes(xes).detach()
        image = values.view(xv.size()[0], xv.size()[1]).t().cpu().numpy()
        if imshow:
            plt.scatter(m.positions[0, :, 0].cpu().numpy(), m.positions[0, :, 1].cpu().numpy(), zorder=1)
            plt.imshow(image, zorder=0, extent=[x_low, x_high, y_low, y_high], origin='lower')
            plt.colorbar()
            plt.show()
        return image

    def cuda(self) -> Mixture:
        return Mixture(self.weights.cuda(), self.positions.cuda(), self.covariances.cuda())

    def cpu(self) -> Mixture:
        return Mixture(self.weights.cpu(), self.positions.cpu(), self.covariances.cpu())

    def to(self, device: torch.device) -> Mixture:
        return Mixture(self.weights.to(device), self.positions.to(device), self.covariances.to(device))

    def batch(self, batch_id: int) -> Mixture:
        n_dims = self.n_dimensions()
        ret_mixture = generate_null_mixture(1, 1, n_dims, device=self.device())
        ret_mixture.weights = self.weights[batch_id, :].view(1, -1)
        ret_mixture.positions = self.positions[batch_id, :, :].view(1, -1, n_dims)
        ret_mixture.covariances = self.covariances[batch_id, :, :, :].view(1, -1, n_dims, n_dims)
        if self._inverted_covariances is not None:
            ret_mixture._inverted_covariances = self._inverted_covariances[batch_id, :, :, :].view(1, -1, n_dims, n_dims)
        return ret_mixture

    def detach(self) -> Mixture:
        detached_mixture = generate_null_mixture(1, 1, self.n_dimensions(), device=self.device())
        detached_mixture.weights = self.weights.detach()
        detached_mixture.positions = self.positions.detach()
        detached_mixture.covariances = self.covariances.detach()
        if self._inverted_covariances is not None:
            detached_mixture._inverted_covariances = self._inverted_covariances.detach()
        return detached_mixture

    def save(self, file_name: str, meta_info=None) -> None:
        dict = {
            "type": "gm.Mixture",
            "version": 3,
            "weights": self.weights.detach().cpu(),
            "positions": self.positions.detach().cpu(),
            "covariances": self.covariances.detach().cpu(),
            "meta_info": meta_info
        }
        torch.save(dict, "/home/madam/temp/prototype/" + file_name)

    @classmethod
    def load(cls, file_name: str) -> Mixture:
        dict = torch.load("/home/madam/temp/prototype/" + file_name)
        assert dict["type"] == "gm.Mixture"
        assert dict["version"] == 3
        return Mixture(dict["weights"], dict["positions"], dict["covariances"]), dict["meta_info"]


class MixtureReLUandBias:
    def __init__(self, mixture: Mixture, bias: Tensor) -> None:
        assert (bias >= 0).all()
        assert bias.size()[0] == mixture.n_layers()
        self.mixture = mixture
        self.bias = bias

    def detach(self):
        return MixtureReLUandBias(self.mixture.detach(), self.bias.detach())

    def evaluate_few_xes(self, positions: Tensor) -> Tensor:
        values = self.mixture.evaluate_few_xes(positions) - self.bias.view(-1, 1)
        return torch.max(values, torch.tensor([0.0001], dtype=torch.float32, device=self.mixture.device()))

    def debug_show(self, batch_i: int = 0, x_low: float = -22, y_low: float = -22, x_high: float = 22, y_high: float = 22, step: float = 0.1, imshow=True) -> Tensor:
        assert self.mixture.n_dimensions() == 2
        assert batch_i < self.mixture.n_layers()
        m = self.mixture.detach().batch(batch_i)
        xv, yv = torch.meshgrid([torch.arange(x_low, x_high, step, dtype=torch.float, device=m.device()),
                                 torch.arange(y_low, y_high, step, dtype=torch.float, device=m.device())])
        xes = torch.cat((xv.reshape(-1, 1), yv.reshape(-1, 1)), 1).view(1, -1, 2)
        values = m.evaluate_many_xes(xes)
        values -= self.bias.detach()[batch_i]
        values[values < 0] = 0
        values = values.view(xv.size()[0], xv.size()[1]).t()
        if imshow:
            image = values.cpu().numpy()
            plt.imshow(image, origin='lower')
            plt.colorbar()
            plt.show()
        return values

    def device(self) -> torch.device:
        return self.mixture.device()


def single_batch_mixture(weights: Tensor, positions: Tensor, covariances: Tensor):
    n_dims = positions.size()[1]
    return Mixture(weights.view(1, -1), positions.view(1, -1, n_dims), covariances.view(1, -1, n_dims, n_dims))


# we will need to work on the initialisation. it's unlikely this simple one will work.
def generate_random_mixtures(n_layers: int, n_components: int, n_dims: int,
                             pos_radius: float = 10,
                             cov_radius: float = 10,
                             weight_min: float = -1,
                             weight_max: float = 1,
                             device: torch.device = 'cpu') -> Mixture:
    assert n_dims == 2 or n_dims == 3
    assert weight_min < weight_max
    assert n_components > 0
    assert n_layers > 0

    weights = torch.rand(n_layers, n_components, dtype=torch.float32, device=device) * (weight_max - weight_min) + weight_min
    positions = torch.rand(n_layers, n_components, n_dims, dtype=torch.float32, device=device) * 2 * pos_radius - pos_radius
    covs = mat_tools.gen_random_positive_definite((n_layers, n_components, n_dims, n_dims), device=device) * cov_radius

    return Mixture(weights, positions, covs)


# todo: this function is a mess
def generate_null_mixture(n_layers: int, n_components: int, n_dims: int, device: torch.device = 'cpu') -> Mixture:
    m = generate_random_mixtures(n_layers, n_components, n_dims, device=device)
    m.weights *= 0
    m.positions *= 0
    m.covariances *= 0
    return m


def _polynomMulRepeat(A: Tensor, B: Tensor) -> (Tensor, Tensor):
    A_n = A.size()[1]
    B_n = B.size()[1]
    A_repeats = [1] * len(A.size())
    A_repeats[1] = B_n
    return (A.repeat(A_repeats), B.repeat_interleave(A_n, dim=1))


def convolve(m1: Mixture, m2: Mixture) -> Mixture:
    n_layers = m1.n_layers()
    n_dims = m1.n_dimensions()
    assert n_layers == m2.n_layers()
    assert n_dims == m2.n_dimensions()
    m1_f, m2_f = _polynomMulRepeat(m1.weights, m2.weights)
    m1_p, m2_p = _polynomMulRepeat(m1.positions, m2.positions)
    # m1_c, m2_c = _polynomMulRepeat(m1.covariances.view(n_layers, m1.n_components(), n_dims * n_dims), m2.covariances.view(n_layers, m2.n_components(), n_dims * n_dims))
    m1_c, m2_c = _polynomMulRepeat(m1.covariances, m2.covariances)
    m1_c = m1_c
    m2_c = m2_c

    positions = m1_p + m2_p
    covariances = m1_c + m2_c
    detc1tc2 = torch.det(m1_c) * torch.det(m2_c)
    detc1pc2 = torch.det(covariances)
    weights = math.pow(math.sqrt(2 * math.pi), m1.n_dimensions()) * m1_f * m2_f * torch.sqrt(detc1tc2) / torch.sqrt(detc1pc2)
    return Mixture(weights, positions, covariances)


def batch_sum(ms: typing.List[Mixture]) -> Mixture:
    assert len(ms) > 0
    weights = []
    positions = []
    covariances = []
    n_dims = ms[0].n_dimensions()
    for m in ms:
        weights.append(m.weights.view(1, -1))
        positions.append(m.positions.view(1, -1, n_dims))
        covariances.append(m.covariances.view(1, -1, n_dims, n_dims))

    return Mixture(torch.cat(weights, dim=0),
                   torch.cat(positions, dim=0),
                   torch.cat(covariances, dim=0))


class NormalisationFactors:
    def __init__(self, weight_scaling: Tensor, position_translation: Tensor, position_scaling: Tensor):
        self.weight_scaling = weight_scaling
        self.position_translation = position_translation
        self.position_scaling = position_scaling


def normalise(data_in: MixtureReLUandBias) -> (MixtureReLUandBias, NormalisationFactors):
    n_layers = data_in.mixture.n_layers()
    n_dims = data_in.mixture.n_dimensions()
    weight_min, _ = torch.min(data_in.mixture.weights.detach(), dim=1)
    weight_max, _ = torch.max(data_in.mixture.weights.detach(), dim=1)
    weight_scaling = torch.max(torch.abs(weight_min), weight_max)
    weight_scaling = torch.max(weight_scaling, data_in.bias.detach())
    weight_scaling = weight_scaling.view(n_layers, 1)
    weight_scaling = (1 / weight_scaling)

    weights_normalised = data_in.mixture.weights * weight_scaling
    bias_normalised = data_in.bias * weight_scaling.view(-1)

    position_translation = (-torch.mean(data_in.mixture.positions.detach(), dim=1)).view(n_layers, 1, n_dims)
    positions_normalised = data_in.mixture.positions + position_translation
    covariance_adjustment =  torch.sqrt(torch.diagonal(data_in.mixture.covariances.detach(), dim1=-2, dim2=-1))
    position_max, _ = torch.max(positions_normalised.detach() + covariance_adjustment, dim=1)
    position_min, _ = torch.min(positions_normalised.detach() - covariance_adjustment, dim=1)
    position_scaling = torch.max(torch.abs(position_min), position_max)
    position_scaling = position_scaling.view(n_layers, 1, n_dims)
    position_scaling = 1 / position_scaling
    positions_normalised *= position_scaling

    covariance_scaling = torch.diag_embed(position_scaling)
    covariances_normalised = covariance_scaling @ data_in.mixture.covariances @ covariance_scaling

    return MixtureReLUandBias(Mixture(weights_normalised, positions_normalised, covariances_normalised), bias_normalised), \
           NormalisationFactors(weight_scaling, position_translation, position_scaling)

def de_normalise(m: Mixture, normalisation: NormalisationFactors) -> Mixture:
    inverted_weight_scaling = 1 / normalisation.weight_scaling
    inverted_position_translation = - normalisation.position_translation
    inverted_position_scaling = 1 / normalisation.position_scaling
    inverted_covariance_scaling = torch.diag_embed(inverted_position_scaling)

    return Mixture(m.weights * inverted_weight_scaling,
                   m.positions * inverted_position_scaling + inverted_position_translation,
                   inverted_covariance_scaling @ m.covariances @ inverted_covariance_scaling)

