from __future__ import annotations
import math
import typing

import torch
from torch import Tensor
import matplotlib.pyplot as plt

import mat_tools
import config


def n_dimensions(mixture: Tensor) -> int:
    vector_length = mixture.shape[-1]
    if vector_length == 7:  # weight: 1, position: 2, covariance: 4
        return 2
    if vector_length == 13:  # weight: 1, position: 3, covariance: 9
        return 3
    print(f"Invalid matrix in gm.n_dims with shape {mixture.shape}!")
    exit(1)


def n_batch(mixture: Tensor) -> int:
    return mixture.shape[0]


def n_layers(mixture: Tensor) -> int:
    return mixture.shape[1]


def n_components(mixture: Tensor) -> int:
    return mixture.shape[2]


def weights(mixture: Tensor) -> Tensor:
    return mixture[:, :, :, 0]


def positions(mixture: Tensor) -> Tensor:
    return mixture[:, :, :, 1:(n_dimensions(mixture) + 1)]


def covariances(mixture: Tensor) -> Tensor:
    _n_dims = n_dimensions(mixture)
    new_shape = list(mixture.shape)
    new_shape[-1] = _n_dims
    new_shape.append(_n_dims)
    return mixture[:, :, :, (_n_dims + 1):].view(new_shape)


def pack_mixture(weights: Tensor, positions: Tensor, covariances: Tensor) -> Tensor:
    assert weights.shape[0] == positions.shape[0] == covariances.shape[0]
    assert weights.shape[1] == positions.shape[1] == covariances.shape[1]
    assert weights.shape[2] == positions.shape[2] == covariances.shape[2]

    weight_shape = list(positions.shape)
    weight_shape[-1] = 1

    cov_shape = list(positions.shape)
    dims = cov_shape[-1]
    assert dims == 2 or dims == 3

    cov_shape[-1] = dims * dims
    return torch.cat((weights.view(weight_shape), positions, covariances.view(cov_shape)), dim=len(positions.shape) - 1)


def is_valid_mixture(mixture: Tensor) -> bool:
    # mixture: 1st dimension: batch, 2nd: layer, 3rd: component, 4th: vector of gaussian data
    ok = True
    ok = ok and len(mixture.shape) == 4
    ok = ok and n_dimensions(mixture) == 2 or n_dimensions(mixture) == 3   # also checks the length of the Gaussian vector
    ok = ok and torch.all(covariances(mixture).det() > 0)
    return ok


def evaluate_few_xes_component_wise(mixture: Tensor, xes: Tensor) -> Tensor:
    _n_batch = n_batch(mixture)
    _n_layers = n_layers(mixture)
    _n_dims = n_dimensions(mixture)
    _n_comps = n_components(mixture)

    # xes dims: 1. batch (may be 1), 2. layers (may be 1), 3. n_xes, 4. x/y/[z]
    assert len(xes.shape) == 4
    assert xes.shape[0] == 1 or xes.shape[0] == _n_batch
    assert xes.shape[1] == 1 or xes.shape[1] == _n_layers
    n_xes = xes.shape[2]
    assert xes.shape[3] == _n_dims

    if _n_batch * _n_layers * _n_comps * n_xes < 100 * 1024 * 1024:
        # 1. dim: batches, 2. layers, 3. component, 4. xes, 5.+: vector / matrix components
        xes = xes.view(xes.shape[0], xes.shape[1], 1, n_xes, _n_dims)
        _positions = positions(mixture).view(_n_batch, _n_layers, _n_comps, 1, _n_dims)
        values = xes - _positions

        # x^t A x -> quadratic form
        x_t = values.view(_n_batch, _n_layers, _n_comps, -1, 1, _n_dims)
        x = values.view(_n_batch, _n_layers, _n_comps, -1, _n_dims, 1)
        A = covariances(mixture).inverse().view(_n_batch, _n_layers, _n_comps, 1, _n_dims, _n_dims)
        values = -0.5 * x_t @ A @ x
        values = values.view(_n_batch, _n_layers, _n_comps, -1)
    else:
        # todo: select min of _n_batch and n_components or something?
        # todo: test
        batched_values = torch.zeros(_n_batch, _n_layers, _n_comps, n_xes, dtype=torch.float32, device=self.device())
        for i in range(_n_batch):
            xes_slice = xes[i, 1, :, :].view(1, 1, -1, _n_dims)
            _positions = positions(mixture)[i, :, :, :].view(_n_layers, _n_comps, 1, _n_dims)
            values = xes_slice - _positions

            # x^t A x -> quadratic form
            x_t = values.view(_n_layers, _n_comps, -1, 1, _n_dims)
            x = values.view(_n_layers, _n_comps, -1, _n_dims, 1)
            A = covariances(mixture)[i, :, :, :, :].inverse().view(_n_layers, _n_comps, 1, _n_dims, _n_dims)
            values = -0.5 * x_t @ A @ x
            batched_values[i, :, :, :] = values.view(_n_layers, _n_comps, -1)
        values = batched_values

    values = weights(mixture).view(_n_batch, _n_layers, _n_comps, 1) * torch.exp(values)
    return values.view(_n_batch, _n_layers, _n_comps, -1)


def evaluate_inversed(mixture: Tensor, xes: Tensor) -> Tensor:
    _n_batch = n_batch(mixture)
    _n_layers = n_layers(mixture)
    _n_dims = n_dimensions(mixture)
    _n_comps = n_components(mixture)

    # xes dims: 1. batch (may be 1), 2. layers (may be 1), 3. n_xes, 4. x/y/[z]
    assert len(xes.shape) == 4
    assert xes.shape[0] == 1 or xes.shape[0] == _n_batch
    assert xes.shape[1] == 1 or xes.shape[1] == _n_layers
    n_xes = xes.shape[2]
    assert xes.shape[3] == _n_dims

    xes = xes.view(xes.shape[0], xes.shape[1], 1, n_xes, _n_dims)
    values_sum = torch.zeros(_n_batch, _n_layers, n_xes, dtype=torch.float32, device=mixture.device)

    total_memory_space = _n_batch * _n_layers * _n_comps * n_xes * _n_dims # did i forget something
    n_memory_slices = max(total_memory_space // (1024 * 1024 * 100), 1)
    comp_slice_size = _n_comps // n_memory_slices
    n_memory_slices = _n_comps // comp_slice_size + int(_n_comps % comp_slice_size != 0)
    for i in range(n_memory_slices):
        # 1. dim: batches, 2. layers, 3. component, 4. xes, 5.+: vector / matrix components
        comps_begin = i * comp_slice_size
        comps_end = min(comps_begin + comp_slice_size, _n_comps)
        n_comps_slice = comps_end - comps_begin
        _positions = positions(mixture)[:, :, comps_begin:comps_end, :].view(_n_batch, _n_layers, n_comps_slice, 1, _n_dims)
        values = xes - _positions

        # x^t A x -> quadratic form
        x_t = values.view(_n_batch, _n_layers, n_comps_slice, -1, 1, _n_dims)
        x = values.view(_n_batch, _n_layers, n_comps_slice, -1, _n_dims, 1)
        A = covariances(mixture)[:, :, comps_begin:comps_end].view(_n_batch, _n_layers, n_comps_slice, 1, _n_dims, _n_dims)
        values = -0.5 * x_t @ A @ x
        values = values.view(_n_batch, _n_layers, n_comps_slice, -1)

        values = weights(mixture)[:, :, comps_begin:comps_end].view(_n_batch, _n_layers, n_comps_slice, 1) * torch.exp(values)
        values_sum += values.sum(dim=2)
    return values_sum


def evaluate(mixture: Tensor, xes: Tensor) -> Tensor:
    # torch inverse returns a transposed matrix (v 1.3.1). our matrix is symmetric however, and we want to take a view, so the transpose avoids a copy.
    return evaluate_inversed(pack_mixture(weights(mixture), positions(mixture), covariances(mixture).inverse().transpose(-2, -1)), xes)


# todo: untested
def max_component(mixture: Tensor, xes: Tensor) -> Tensor:
    assert n_layers(mixture) == 1
    selected = torch.zeros(xes.size()[1], dtype=torch.long)
    component_values = evaluate_few_xes_component_wise(mixture, xes)
    values = component_values[:, :, 0]

    for i in range(n_components(mixture)):
        component_values = component_values[:, :, 0]
        mask = component_values > values
        selected[mask] = i
        values[mask] = component_values[mask]

    return selected


def debug_show(mixture: Tensor, batch_i: int = 0, layer_i: int = 0, x_low: float = -22, y_low: float = -22, x_high: float = 22, y_high: float = 22, step: float = 0.1, imshow=True) -> Tensor:
    assert n_dimensions(mixture) == 2
    assert batch_i < n_batch(mixture)
    assert layer_i < n_layers(mixture)
    assert is_valid_mixture(mixture)
    m = mixture.detach()[batch_i, layer_i].view(1, 1, n_components(mixture), -1)

    xv, yv = torch.meshgrid([torch.arange(x_low, x_high, step, dtype=torch.float, device=mixture.device),
                             torch.arange(y_low, y_high, step, dtype=torch.float, device=mixture.device)])
    xes = torch.cat((xv.reshape(-1, 1), yv.reshape(-1, 1)), 1).view(1, 1, -1, 2)

    values = evaluate(m, xes).detach()
    image = values.view(xv.size()[0], xv.size()[1]).t().cpu().numpy()
    if imshow:
        plt.scatter(positions(m)[0, 0, :, 0].cpu().numpy(), positions(m)[0, 0, :, 1].cpu().numpy(), zorder=1)
        plt.imshow(image, zorder=0, extent=[x_low, x_high, y_low, y_high], origin='lower')
        plt.colorbar()
        plt.show()
    return image


def save(mixture: Tensor, file_name: str, meta_info=None) -> None:
    dict = {
        "type": "gm.Mixture",
        "version": 5,
        "data": mixture.detach().cpu(),
        "meta_info": meta_info
    }
    torch.save(dict, config.data_base_path / file_name)


def load(file_name: str) -> typing.Tuple[Tensor, typing.Any]:
    dictionary = torch.load(config.data_base_path / file_name)
    assert dictionary["type"] == "gm.Mixture"
    if dictionary["version"] == 3:
        weights = dictionary["weights"]
        n_components = weights.shape[1]
        positions = dictionary["positions"]
        n_dims = positions.shape[2]
        covariances = dictionary["covariances"]
        mixture = pack_mixture(weights.view(-1, 1, n_components), positions.view(-1, 1, n_components, n_dims), covariances.view(-1, 1, n_components, n_dims, n_dims)), dictionary["meta_info"]

    elif dictionary["version"] == 4:
        weights = dictionary["weights"]
        positions = dictionary["positions"]
        covariances = dictionary["covariances"]
        mixture = pack_mixture(weights, positions, covariances), dictionary["meta_info"]
    else:
        assert dictionary["version"] == 5
        mixture = dictionary["data"], dictionary["meta_info"]

    assert is_valid_mixture(mixture)
    return mixture


def is_valid_mixture_and_bias(mixture: Tensor, bias: Tensor) -> bool:
    ok = True
    ok = ok and (bias >= 0).all()
    ok = ok and is_valid_mixture(mixture)
    ok = ok and len(bias.shape) == 2
    ok = ok and (bias.shape[0] == 1 or bias.shape[0] == mixture[0])
    ok = ok and bias.shape[1] == mixture[1]
    ok = ok and mixture.device == bias.device
    return ok


def evaluate_with_activation_fun(mixture: Tensor, bias: Tensor, xes: Tensor) -> Tensor:
    assert is_valid_mixture_and_bias(mixture, bias)
    bias_shape = list(bias.shape)
    bias_shape.append(1)
    values = evaluate(mixture, xes) - bias.view(bias_shape)
    return torch.max(values, torch.tensor([0.00001], dtype=torch.float32, device=mixture.device))


def debug_show_with_activation_fun(mixture: Tensor, bias: Tensor, batch_i: int = 0, layer_i: int = 0, x_low: float = -22, y_low: float = -22, x_high: float = 22, y_high: float = 22, step: float = 0.1, imshow=True) -> Tensor:
    assert is_valid_mixture_and_bias(mixture, bias)
    assert n_dimensions(mixture) == 2
    assert batch_i < n_batch(mixture)
    assert layer_i < n_layers(mixture)

    mixture_shape = mixture.shape
    mixture_shape[0] = 1
    mixture_shape[1] = 1
    mixture = mixture.detach()[batch_i][layer_i].view(mixture_shape)

    xv, yv = torch.meshgrid([torch.arange(x_low, x_high, step, dtype=torch.float, device=m.device()),
                             torch.arange(y_low, y_high, step, dtype=torch.float, device=m.device())])
    xes = torch.cat((xv.reshape(-1, 1), yv.reshape(-1, 1)), 1).view(1, -1, 2)
    values = evaluate_with_activation_fun(mixture, bias, xes)
    values = values.view(xv.size()[0], xv.size()[1]).t()
    if imshow:
        image = values.cpu().numpy()
        plt.imshow(image, origin='lower')
        plt.colorbar()
        plt.show()
    return values


# we will need to work on the initialisation. it's unlikely this simple one will work.
def generate_random_mixtures(n_batch: int = 1, n_layers: int = 1, n_components: int = 1, n_dims: int = 2,
                             pos_radius: float = 10,
                             cov_radius: float = 10,
                             weight_min: float = -1,
                             weight_max: float = 1,
                             device: torch.device = 'cpu') -> Tensor:
    assert n_batch > 0
    assert n_layers > 0
    assert n_components > 0
    assert n_dims == 2 or n_dims == 3
    assert weight_min < weight_max

    weights = torch.rand(n_batch, n_layers, n_components, dtype=torch.float32, device=device) * (weight_max - weight_min) + weight_min
    positions = torch.rand(n_batch, n_layers, n_components, n_dims, dtype=torch.float32, device=device) * 2 * pos_radius - pos_radius
    covs = mat_tools.gen_random_positive_definite((n_batch, n_layers, n_components, n_dims, n_dims), device=device) * cov_radius

    return pack_mixture(weights, positions, covs)


def _polynomMulRepeat(A: Tensor, B: Tensor) -> (Tensor, Tensor):
    A_n = A.shape[2]
    B_n = B.shape[2]
    A_repeats = [1] * len(A.shape)
    A_repeats[2] = B_n
    return (A.repeat(A_repeats), B.repeat_interleave(A_n, dim=2))


def convolve(m1: Tensor, m2: Tensor) -> Tensor:
    assert n_batch(m1) == 1 or n_batch(m2) == 1 or n_batch(m1) == n_batch(m2)
    assert n_layers(m1) == n_layers(m2)
    assert n_dimensions(m1) == n_dimensions(m2)
    m1, m2 = _polynomMulRepeat(m1, m2)

    m1_c = covariances(m1)
    m2_c = covariances(m2)
    m_new = m1 + m2
    m_new_c = covariances(m_new)
    detc1tc2 = torch.det(m1_c) * torch.det(m2_c)
    detc1pc2 = torch.det(m_new_c)
    m1_w = weights(m1)
    m2_w = weights(m2)
    m_new_w = math.pow(math.sqrt(2 * math.pi), n_dimensions(m1)) * m1_w * m2_w * torch.sqrt(detc1tc2) / torch.sqrt(detc1pc2)
    m_new[:, :, :, 0] = m_new_w

    assert is_valid_mixture(m_new)
    return m_new


class NormalisationFactors:
    def __init__(self, weight_scaling: Tensor, position_translation: Tensor, position_scaling: Tensor):
        self.weight_scaling = weight_scaling
        self.position_translation = position_translation
        self.position_scaling = position_scaling


def normalise(mixture_in: Tensor, bias_in: Tensor) -> (Tensor, Tensor, NormalisationFactors):
    _n_batch = n_batch(mixture_in)
    _n_layers = n_layers(mixture_in)
    _n_dims = n_dimensions(mixture_in)
    weight_min, _ = torch.min(weights(mixture_in.detach()), dim=2)
    weight_max, _ = torch.max(weights(mixture_in.detach()), dim=2)
    weight_scaling = torch.max(torch.abs(weight_min), weight_max)
    weight_scaling = torch.max(weight_scaling, bias_in.detach())
    weight_scaling = weight_scaling.view(_n_batch, _n_layers, 1)
    weight_scaling = torch.ones(1, dtype=torch.float32, device=mixture_in.device) / weight_scaling

    weights_normalised = weights(mixture_in) * weight_scaling
    bias_normalised = bias_in.view(1, _n_layers) * weight_scaling.view(_n_batch, _n_layers)

    position_translation = (-torch.mean(positions(mixture_in.detach()), dim=2)).view(_n_batch, _n_layers, 1, _n_dims)
    positions_normalised = positions(mixture_in) + position_translation
    covariance_adjustment = torch.sqrt(torch.diagonal(covariances(mixture_in.detach()), dim1=-2, dim2=-1))
    position_max, _ = torch.max(positions_normalised.detach() + covariance_adjustment, dim=2)
    position_min, _ = torch.min(positions_normalised.detach() - covariance_adjustment, dim=2)
    position_scaling = torch.max(torch.abs(position_min), position_max)
    position_scaling = position_scaling.view(_n_batch, _n_layers, 1, _n_dims)
    position_scaling = torch.ones(1, dtype=torch.float32, device=mixture_in.device) / position_scaling
    positions_normalised *= position_scaling

    covariance_scaling = torch.diag_embed(position_scaling)
    covariances_normalised = covariance_scaling @ covariances(mixture_in) @ covariance_scaling

    return pack_mixture(weights_normalised, positions_normalised, covariances_normalised), bias_normalised, NormalisationFactors(weight_scaling, position_translation, position_scaling)


def de_normalise(m: Tensor, normalisation: NormalisationFactors) -> Tensor:
    inverted_weight_scaling = torch.ones(1, dtype=torch.float32, device=m.device) / normalisation.weight_scaling
    inverted_position_translation = - normalisation.position_translation
    inverted_position_scaling = torch.ones(1, dtype=torch.float32, device=m.device) / normalisation.position_scaling
    inverted_covariance_scaling = torch.diag_embed(inverted_position_scaling)

    return pack_mixture(weights(m) * inverted_weight_scaling,
                        positions(m) * inverted_position_scaling + inverted_position_translation,
                        inverted_covariance_scaling @ covariances(m) @ inverted_covariance_scaling)
