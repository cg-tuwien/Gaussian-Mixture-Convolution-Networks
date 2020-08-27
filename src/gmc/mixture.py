#from __future__ import annotations
import math
import typing
import os

import torch
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np

from . import mat_tools
from . import config
from .cpp.extensions.evaluate import evaluate_inversed as gm_evaluate_inversed


def n_dimensions(mixture: Tensor) -> int:
    vector_length = mixture.shape[-1]
    if vector_length == 7:  # weight: 1, position: 2, covariance: 4
        return 2
    if vector_length == 13:  # weight: 1, position: 3, covariance: 9
        return 3
    print(f"Invalid matrix in gm.n_dims with shape {mixture.shape}!")
    assert False
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
    if not weights.shape[1] == positions.shape[1] == covariances.shape[1]:
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
    # # mixture: 1st dimension: batch, 2nd: layer, 3rd: component, 4th: vector of gaussian data
    # ok = True
    # ok = ok and len(mixture.shape) == 4
    # ok = ok and n_dimensions(mixture) == 2 or n_dimensions(mixture) == 3   # also checks the length of the Gaussian vector
    # ok = ok and torch.all(covariances(mixture).det() > 0)
    # return ok

    # mixture: 1st dimension: batch, 2nd: layer, 3rd: component, 4th: vector of gaussian data
    assert not torch.any(torch.isnan(mixture))
    assert not torch.any(torch.isinf(mixture))
    assert len(mixture.shape) == 4
    assert n_dimensions(mixture) == 2 or n_dimensions(mixture) == 3   # also checks the length of the Gaussian vector
    assert torch.all(covariances(mixture).det() > 0)
    return True


def integrate_components(mixture: Tensor) -> Tensor:
    assert is_valid_mixture(mixture)
    dets = torch.det(covariances(mixture))
    return weights(mixture) * torch.sqrt((2 * math.pi) ** n_dimensions(mixture) * dets)


def integrate(mixture: Tensor) -> Tensor:
    return integrate_components(mixture).sum(dim=2)


# returns the amplitude of a multivariate normal distribution with the given covariance
# version for when the covariance is already inversed (det(C)^-1 == det(C^-1))
def normal_amplitudes_inversed(_covariances: Tensor) -> Tensor:
    n_dims = _covariances.shape[-1]
    return (2 * math.pi) ** (- n_dims * 0.5) * torch.sqrt(torch.det(_covariances))


# returns the amplitude of a multivariate normal distribution with the given covariance
# version for when the covariance is in the normal form (det(C)^-1 == det(C^-1))
def normal_amplitudes(_covariances: Tensor) -> Tensor:
    n_dims = _covariances.shape[-1]
    return (2 * math.pi) ** (- n_dims * 0.5) / torch.sqrt(torch.det(_covariances))


def evaluate_inversed(mixture: Tensor, xes: Tensor) -> Tensor:
    return gm_evaluate_inversed.apply(mixture, xes)


def old_evaluate_inversed(mixture: Tensor, xes: Tensor) -> Tensor:
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

    total_memory_space = _n_batch * _n_layers * _n_comps * n_xes * _n_dims * 4  # did i forget something?
    n_memory_slices = max(total_memory_space // config.eval_slize_size, 1) # SLICING
    comp_slice_size = max(_n_comps // n_memory_slices, 1)
    n_memory_slices = _n_comps // comp_slice_size + int(_n_comps % comp_slice_size != 0)
    for i in range(n_memory_slices):
        # 1. dim: batches, 2. layers, 3. component, 4. xes, 5.+: vector / matrix components
        comps_begin = i * comp_slice_size
        comps_end = min(comps_begin + comp_slice_size, _n_comps)
        n_comps_slice = comps_end - comps_begin
        mixture_slice = mixture[:, :, comps_begin:comps_end, :]
        _positions = positions(mixture_slice).view(_n_batch, _n_layers, n_comps_slice, 1, _n_dims)
        values = xes - _positions

        # x^t A x -> quadratic form
        x_t = values.view(_n_batch, _n_layers, n_comps_slice, -1, 1, _n_dims)
        x = values.view(_n_batch, _n_layers, n_comps_slice, -1, _n_dims, 1)
        A = covariances(mixture_slice).view(_n_batch, _n_layers, n_comps_slice, 1, _n_dims, _n_dims)
        values = -0.5 * x_t @ A @ x # 0.8 -> 3.0gb
        values = values.view(_n_batch, _n_layers, n_comps_slice, -1)

        values = weights(mixture_slice).view(_n_batch, _n_layers, n_comps_slice, 1) * torch.exp(values) # 3.0 -> 3.3Gb
        values_sum += values.sum(dim=2)
    return values_sum


def evaluate(mixture: Tensor, xes: Tensor) -> Tensor:
    # torch inverse returns a transposed matrix (v 1.3.1). our matrix is symmetric however, and we want to take a view, so the transpose avoids a copy.
    return evaluate_inversed(pack_mixture(weights(mixture), positions(mixture), covariances(mixture).inverse().transpose(-2, -1)), xes)


def evaluate_componentwise_inversed(gaussians: Tensor, xes: Tensor):
    _n_batch = n_batch(gaussians)
    _n_layers = n_layers(gaussians)
    _n_dims = n_dimensions(gaussians)
    _n_comps = n_components(gaussians)

    # xes dims: 1. batch (may be 1), 2. layers (may be 1), 3. n_xes, 4. x/y/[z]
    assert len(xes.shape) == 4
    assert xes.shape[0] == 1 or xes.shape[0] == _n_batch
    assert xes.shape[1] == 1 or xes.shape[1] == _n_layers
    n_xes = xes.shape[2]
    assert xes.shape[3] == _n_dims

    xes = xes.view(xes.shape[0], xes.shape[1], n_xes, 1, _n_dims)

    # 1. dim: batches, 2. layers, 3. xes, 4. component, 5.+: vector / matrix components
    _positions = positions(gaussians).view(_n_batch, _n_layers, 1, _n_comps, _n_dims)
    values = xes - _positions

    # x^t A x -> quadratic form
    x_t = values.view(_n_batch, _n_layers, n_xes, _n_comps, 1, _n_dims)
    x = values.view(_n_batch, _n_layers, n_xes, _n_comps, _n_dims, 1)
    A = covariances(gaussians).view(_n_batch, _n_layers, 1, _n_comps, _n_dims, _n_dims)
    values = -0.5 * x_t @ A @ x  # 0.8 -> 3.0gb
    values = values.view(_n_batch, _n_layers, n_xes, _n_comps)

    values = weights(gaussians).view(_n_batch, _n_layers, 1, _n_comps) * torch.exp(values)
    return values


def evaluate_componentwise(mixture: Tensor, xes: Tensor) -> Tensor:
    # torch inverse returns a transposed matrix (v 1.3.1). our matrix is symmetric however, and we want to take a view, so the transpose avoids a copy.
    return evaluate_componentwise_inversed(pack_mixture(weights(mixture), positions(mixture), covariances(mixture).inverse().transpose(-2, -1)), xes)


# # todo: untested
# def max_component(mixture: Tensor, xes: Tensor) -> Tensor:
#     assert n_layers(mixture) == 1
#     selected = torch.zeros(xes.size()[1], dtype=torch.long)
#     component_values = evaluate_few_xes_component_wise(mixture, xes)
#     values = component_values[:, :, 0]
#
#     for i in range(n_components(mixture)):
#         component_values = component_values[:, :, 0]
#         mask = component_values > values
#         selected[mask] = i
#         values[mask] = component_values[mask]
#
#     return selected


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
        # plt.scatter(positions(m)[0, 0, :, 0].cpu().numpy(), y_high + y_low - positions(m)[0, 0, :, 1].cpu().numpy(), zorder=1)
        plt.imshow(image, zorder=0, extent=[x_low, x_high, y_low, y_high])
        plt.colorbar()
        plt.show()
    return image


def render(mixture: Tensor, constant: Tensor, batches: typing.Tuple[int, int] = (0, None), layers: typing.Tuple[int, int] = (0, None),
           x_low: float = -22, y_low: float = -22, x_high: float = 22, y_high: float = 22,
           width: int = 100, height: int = 100):
    assert n_dimensions(mixture) == 2
    assert is_valid_mixture(mixture)
    xv, yv = torch.meshgrid([torch.arange(x_low, x_high, (x_high - x_low) / width, dtype=torch.float, device=mixture.device),
                             torch.arange(y_low, y_high, (y_high - y_low) / height, dtype=torch.float, device=mixture.device)])
    m = mixture.detach()[batches[0]:batches[1], layers[0]:layers[1]]
    c = constant.detach()[batches[0]:batches[1], layers[0]:layers[1]]
    n_batch = m.shape[0]
    n_layers = m.shape[1]
    xes = torch.cat((xv.reshape(-1, 1), yv.reshape(-1, 1)), 1).view(1, 1, -1, 2)
    rendering = (evaluate(m, xes) + c.unsqueeze(-1)).view(n_batch, n_layers, width, height).transpose(2, 3)
    rendering = rendering.transpose(0, 1).reshape(n_layers * height, n_batch * width)
    return rendering


def render_with_relu(mixture: Tensor, constant: Tensor,
                     batches: typing.Tuple[int, int] = (0, None), layers: typing.Tuple[int, int] = (0, None),
                     x_low: float = -22, y_low: float = -22, x_high: float = 22, y_high: float = 22,
                     width: int = 100, height: int = 100) -> Tensor:
    assert is_valid_mixture_and_constant(mixture, constant)
    rendering = render(mixture, constant, batches, layers, x_low, y_low, x_high, y_high, width, height)
    return torch.max(rendering, torch.tensor([0.00001], dtype=torch.float32, device=mixture.device))


def export_as_image(mixture: Tensor) -> None:
    # todo make general on next occasion
    rendering = render(mixture.detach().view(1, -1, 1, 7), x_low=-1.5, x_high=1.5, y_low=-1.5, y_high=1.5).cpu().numpy()
    import gmc.image_tools
    rendering_cm = gmc.image_tools.colour_mapped(rendering, -1, 1)
    rendering_cm = rendering_cm.reshape(-1, 100, 100, 4)
    for i in range(rendering_cm.shape[0]):
        plt.imsave(f"{i:05}.png", rendering_cm[i, :, :, :])


def save(mixture: Tensor, file_name: str, meta_info=None) -> None:
    assert is_valid_mixture(mixture)
    dictionary = {
        "type": "gm.Mixture",
        "version": 5,
        "data": mixture.detach().cpu(),
        "meta_info": meta_info
    }
    torch.save(dictionary, config.data_base_path / file_name)


def load(file_name: str) -> typing.Tuple[Tensor, typing.Any]:
    dictionary = torch.load(config.data_base_path / file_name)
    assert dictionary["type"] == "gm.Mixture"
    if dictionary["version"] == 3:
        weights = dictionary["weights"]
        n_components = weights.shape[1]
        positions = dictionary["positions"]
        n_dims = positions.shape[2]
        covariances = dictionary["covariances"]
        mixture = pack_mixture(weights.view(-1, 1, n_components), positions.view(-1, 1, n_components, n_dims), covariances.view(-1, 1, n_components, n_dims, n_dims))

    elif dictionary["version"] == 4:
        weights = dictionary["weights"]
        positions = dictionary["positions"]
        covariances = dictionary["covariances"]
        mixture = pack_mixture(weights, positions, covariances)
    else:
        assert dictionary["version"] == 5
        mixture = dictionary["data"]

    assert is_valid_mixture(mixture)
    return mixture, dictionary["meta_info"]


def is_valid_mixture_and_constant(mixture: Tensor, constant: Tensor) -> bool:
    # ok = True
    # ok = ok and is_valid_mixture(mixture)
    # # t o d o : actually, i think the batch dimension is not needed for the constant
    # ok = ok and len(constant.shape) == 2
    # ok = ok and (constant.shape[0] == 1 or constant.shape[0] == mixture.shape[0])
    # ok = ok and constant.shape[1] == mixture.shape[1]
    # ok = ok and mixture.device == constant.device
    # return ok

    assert is_valid_mixture(mixture)
    # todo: actually, i think the batch dimension is not needed for the constant
    assert len(constant.shape) == 2
    assert constant.shape[0] == 1 or constant.shape[0] == mixture.shape[0]
    assert constant.shape[1] == 1 or constant.shape[1] == mixture.shape[1]
    assert mixture.device == constant.device
    return True


def evaluate_with_activation_fun(mixture: Tensor, bias: Tensor, xes: Tensor) -> Tensor:
    assert is_valid_mixture_and_constant(mixture, bias)
    bias_shape = list(bias.shape)
    bias_shape.append(1)
    values = evaluate(mixture, xes) + bias.view(bias_shape)
    return torch.max(values, torch.tensor([0.00001], dtype=torch.float32, device=mixture.device))


def debug_show_with_activation_fun(mixture: Tensor, bias: Tensor, batch_i: int = 0, layer_i: int = 0, x_low: float = -22, y_low: float = -22, x_high: float = 22, y_high: float = 22, step: float = 0.1, imshow=True) -> Tensor:
    assert is_valid_mixture_and_constant(mixture, bias)
    assert n_dimensions(mixture) == 2
    assert batch_i < n_batch(mixture)
    assert layer_i < n_layers(mixture)

    mixture_shape = list(mixture.shape)
    mixture_shape[0] = 1
    mixture_shape[1] = 1
    mixture = mixture.detach()[batch_i][layer_i].view(mixture_shape)

    bias_shape = list(bias.shape)
    bias_shape[0] = 1
    bias_shape[1] = 1
    bias_batch_i = 0 if bias.shape[0] == 1 else batch_i
    bias_layer_i = 0 if bias.shape[1] == 1 else layer_i
    bias = bias.detach()[bias_batch_i][bias_layer_i].view(bias_shape)

    xv, yv = torch.meshgrid([torch.arange(x_low, x_high, step, dtype=torch.float, device=mixture.device),
                             torch.arange(y_low, y_high, step, dtype=torch.float, device=mixture.device)])
    xes = torch.cat((xv.reshape(-1, 1), yv.reshape(-1, 1)), 1).view(1, 1, -1, 2)
    values = evaluate_with_activation_fun(mixture, bias, xes)
    values = values.view(xv.size()[0], xv.size()[1]).t()
    if imshow:
        image = values.cpu().numpy()
        plt.imshow(image)
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
    assert is_valid_mixture(m1)
    assert is_valid_mixture(m2)
    m1, m2 = _polynomMulRepeat(m1, m2)

    m1_c = covariances(m1)
    m2_c = covariances(m2)
    m_new = m1 + m2
    m_new_c = covariances(m_new)
    assert (torch.det(m_new_c) > 0).all()
    detc1tc2 = torch.det(m1_c) * torch.det(m2_c)
    detc1pc2 = torch.det(m_new_c)
    m1_w = weights(m1)
    m2_w = weights(m2)
    assert not torch.isnan(m1_w).any()
    assert not torch.isnan(m2_w).any()
    assert not torch.isnan(detc1tc2).any()
    assert not torch.isnan(detc1pc2).any()
    assert not torch.isnan(torch.sqrt(detc1tc2)).any()
    assert not torch.isnan(torch.sqrt(detc1pc2)).any()
    m_new_w = math.pow(math.sqrt(2 * math.pi), n_dimensions(m1)) * m1_w * m2_w * torch.sqrt(detc1tc2) / torch.sqrt(detc1pc2)
    assert not torch.isnan(m_new_w).any()
    m_new_w = m_new_w.view(n_batch(m_new), n_layers(m_new), n_components(m_new), 1)
    assert not torch.isnan(m_new_w).any()
    m_new = m_new[:, :, :, 1:]
    m_new = torch.cat((m_new_w, m_new), dim=-1)

    assert is_valid_mixture(m_new)
    return m_new


class NormalisationFactors:
    def __init__(self, weight_scaling: Tensor, position_translation: Tensor, position_scaling: Tensor):
        self.weight_scaling = weight_scaling
        self.position_translation = position_translation
        self.position_scaling = position_scaling


def normalise(mixture_in: Tensor, bias_in: Tensor) -> (Tensor, Tensor, NormalisationFactors):
    _n_batch = n_batch(mixture_in)
    assert bias_in.shape[0] == 1 or bias_in.shape[0] == _n_batch
    _n_layers = n_layers(mixture_in)
    _n_dims = n_dimensions(mixture_in)
    weight_min, _ = torch.min(weights(mixture_in.detach()), dim=2)
    weight_max, _ = torch.max(weights(mixture_in.detach()), dim=2)
    weight_scaling = torch.max(torch.abs(weight_min), weight_max)
    weight_scaling = torch.max(weight_scaling, bias_in.detach().abs())
    weight_scaling = weight_scaling.view(_n_batch, _n_layers, 1)
    weight_scaling = torch.ones(1, dtype=torch.float32, device=mixture_in.device) / weight_scaling

    weights_normalised = weights(mixture_in) * weight_scaling
    bias_normalised = bias_in.view(bias_in.shape[0], bias_in.shape[1]) * weight_scaling.view(_n_batch, _n_layers)

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


def write_gm_to_ply(weights: Tensor, positions: Tensor, covariances: Tensor, batch: int, filename: str):
    weight_shape = weights.shape #should be (m,1,n)
    pos_shape = positions.shape #should be (m,1,n,3)
    cov_shape = covariances.shape #should be (m,1,n,3,3)
    assert len(weight_shape) == 3
    assert len(pos_shape) == 4
    assert len(cov_shape) == 5
    assert weight_shape[0] == pos_shape[0] == cov_shape[0]
    assert weight_shape[1] == pos_shape[1] == cov_shape[1] == 1
    assert weight_shape[2] == pos_shape[2] == cov_shape[2]
    assert pos_shape[3] == cov_shape[3] == 3
    assert cov_shape[4] == 3
    n = weight_shape[2]

    _weights = weights[batch,0,:].view(n)
    _positions = positions[batch,0,:,:].view(n,3)
    _covs = covariances[batch,0,:,:,:].view(n,3,3)

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    file = open(filename, "w+")
    file.write("ply\nformat ascii 1.0\n")
    file.write(f"element component {n}\n")
    file.write("property float x\nproperty float y\nproperty float z\n")
    file.write("property float covxx\nproperty float covxy\nproperty float covxz\n")
    file.write("property float covyy\nproperty float covyz\nproperty float covzz\n")
    file.write("property float weight\nend_header\n")

    file.close()
    file = open(filename, "ab") #open in append mode

    data = torch.zeros(n, 10)
    data[:, 0:3] = _positions
    data[:, 3] = _covs[:, 0, 0]
    data[:, 4] = _covs[:, 0, 1]
    data[:, 5] = _covs[:, 0, 2]
    data[:, 6] = _covs[:, 1, 1]
    data[:, 7] = _covs[:, 1, 2]
    data[:, 8] = _covs[:, 2, 2]
    data[:, 9] = _weights

    np.savetxt(file, data.detach().numpy(), delimiter="  ")

    # for i in range(0,n):
    #     file.write(f"{_positions[i,0].item()}  {_positions[i,1].item()}  {_positions[i,2].item()}  ")
    #     file.write(f"{_covs[i,0,0].item()}  {_covs[i,0,1].item()}  {_covs[i,0,2].item()}  ")
    #     file.write(f"{_covs[i,1,1].item()}  {_covs[i,1,2].item()}  {_covs[i,2,2].item()}  ")
    #     file.write(f"{_weights[i].item()}\n")
    file.close()


def read_gm_from_ply(filename: str, ismodel: bool) -> Tensor:
    #THIS IS VERY SIMPLE AND NOT GENERAL
    fin = open(filename)
    header = True
    index = 0
    for line in fin:
        if header:
            if line.startswith("element component "):
                number = int(line[18:])
                gmpos = torch.zeros((1, 1, number, 3))
                gmcov = torch.zeros((1, 1, number, 3, 3))
                gmwei = torch.zeros((1, 1, number))
            elif line.startswith("end_header"):
                header = False
        else:
            elements = line.split("  ")
            gmpos[0, 0, index, :] = torch.tensor([float(e) for e in elements[0:3]])
            gmcov[0, 0, index, 0, 0] = float(elements[3])
            gmcov[0, 0, index, 0, 1] = gmcov[0, 0, index, 1, 0] = float(elements[4])
            gmcov[0, 0, index, 0, 2] = gmcov[0, 0, index, 2, 0] = float(elements[5])
            gmcov[0, 0, index, 1, 1] = float(elements[6])
            gmcov[0, 0, index, 1, 2] = gmcov[0, 0, index, 2, 1] = float(elements[7])
            gmcov[0, 0, index, 2, 2] = float(elements[8])
            gmwei[0, 0, index] = float(elements[9])
            index = index + 1
    fin.close()
    if ismodel:
        gmwei /= gmwei.sum()
        amplitudes = gmwei / (gmcov.det().sqrt() * 15.74960995)
        return pack_mixture(amplitudes, gmpos, gmcov)
    else:
        return pack_mixture(gmwei, gmpos, gmcov)
