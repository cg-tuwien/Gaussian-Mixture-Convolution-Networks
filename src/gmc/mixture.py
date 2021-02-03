import math

import torch
from torch import Tensor

from . import mat_tools
from . import config
from .cpp.extensions.evaluate_inversed import evaluate_inversed as cppExtensionsEvaluateInversed


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
    return weights(mixture) * torch.sqrt(((2 * math.pi) ** n_dimensions(mixture)) * dets)


def integrate(mixture: Tensor) -> Tensor:
    ## test the cpp version, but disable for now because there is no backward.
    ## we want to use gaussian integrals internally in the cpp fitting code, and had to verify. python integration is not too slow for the time being.
    #return cppExtensionsIntegrateInversed.apply(pack_mixture(weights(mixture), positions(mixture), mat_tools.inverse(covariances(mixture))))
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
    return cppExtensionsEvaluateInversed.apply(mixture, xes)


def evaluate_inversed_bvh(mixture: Tensor, xes: Tensor) -> Tensor:
    return cppExtensionsEvaluateInversed.apply_bvh(mixture, xes)


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
    return evaluate_inversed(pack_mixture(weights(mixture), positions(mixture), mat_tools.inverse(covariances(mixture)).transpose(-1, -2)), xes)


def evaluate_bvh(mixture: Tensor, xes: Tensor) -> Tensor:
    # torch inverse returns a transposed matrix (v 1.3.1). our matrix is symmetric however, and we want to take a view, so the transpose avoids a copy.
    return evaluate_inversed_bvh(pack_mixture(weights(mixture), positions(mixture), mat_tools.inverse(covariances(mixture)).transpose(-1, -2)), xes)


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


def _polynomial_mul_repeat(A: Tensor, B: Tensor) -> (Tensor, Tensor):
    A_n = A.shape[2]
    B_n = B.shape[2]
    A_repeats = [1] * len(A.shape)
    A_repeats[2] = B_n
    return A.repeat(A_repeats), B.repeat_interleave(A_n, dim=2)


def convolve(m1: Tensor, m2: Tensor) -> Tensor:
    assert n_batch(m1) == 1 or n_batch(m2) == 1 or n_batch(m1) == n_batch(m2)
    assert n_layers(m1) == n_layers(m2)
    assert n_dimensions(m1) == n_dimensions(m2)
    assert is_valid_mixture(m1)
    assert is_valid_mixture(m2)
    m1, m2 = _polynomial_mul_repeat(m1, m2)

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


def spatial_scale(m: Tensor, scaling_factors: Tensor) -> Tensor:
    """Does not scale weights, i.e., the integral will change."""
    scaling_factors = scaling_factors.view(n_batch(m), n_layers(m), 1, n_dimensions(m))
    w = weights(m)
    p = positions(m) * scaling_factors
    cs = torch.diag_embed(scaling_factors)
    c = cs @ covariances(m) @ cs
    return pack_mixture(w, p, c)


def convert_priors_to_amplitudes(gm: torch.Tensor) -> torch.Tensor:
    # Given mixture has priors as weights
    # This returns a new mixture with corresponding amplitudes as weights
    # This implementation specificly only works for 3D
    gmwei = weights(gm)
    gmcov = covariances(gm)
    amplitudes = gmwei / (gmcov.det().sqrt() * 15.74960995)
    return pack_mixture(amplitudes, positions(gm), gmcov)


def convert_amplitudes_to_priors(gm: torch.Tensor) -> torch.Tensor:
    # Given mixture has amplitudes as weights
    # This returns a new mixture with corresponding priors as weights
    # This implementation specificly only works for 3D
    gmamp = weights(gm)
    gmcov = covariances(gm)
    priors = gmamp * (gmcov.det().sqrt() * 15.74960995)
    return pack_mixture(priors, positions(gm), gmcov)
