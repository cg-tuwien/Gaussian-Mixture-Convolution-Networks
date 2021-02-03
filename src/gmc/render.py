import typing
import matplotlib.cm
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import Tensor

import gmc.mixture as gm
import gmc.cpp.gm_vis.gm_vis as gm_vis

from gmc import colourmap

gamma_correction = 1/2.2

index_t = typing.Optional[int]
index_range = typing.Tuple[index_t, index_t]

def colour_mapped(mono, low, high):
    if mono.ndim > 2:
        raise Exception("colour_map is only applicable for mono matrices")

    normaliser = matplotlib.colors.Normalize(vmin=low, vmax=high, clip=True)
    mapper = matplotlib.cm.ScalarMappable(norm=normaliser, cmap=colourmap.cm_linSeg)
    return mapper.to_rgba(mono)


def write_colour_map(width: int, height: int, filename: str):
    size_large = max(width, height)
    size_small = min(width, height)
    vals = np.linspace(0, 1, size_large)
    vals = colour_mapped(vals, 0, 1)[:, :3]
    if width > height:
        vals = np.reshape(vals, (1, width, 3))
    else:
        vals = np.reshape(vals, (height, 1, 3))

    vals = np.repeat(vals, size_small, axis=int(width < height))
    plt.imsave(filename, vals)


def render(mixture: Tensor, constant: Tensor, batches: index_range = (0, None), layers: index_range = (0, None),
           x_low: float = -22, y_low: float = -22, x_high: float = 22, y_high: float = 22,
           width: int = 100, height: int = 100):
    assert gm.n_dimensions(mixture) == 2
    assert gm.is_valid_mixture(mixture)
    xv, yv = torch.meshgrid([torch.arange(x_low, x_high, (x_high - x_low) / width, dtype=torch.float, device=mixture.device),
                             torch.arange(y_low, y_high, (y_high - y_low) / height, dtype=torch.float, device=mixture.device)])
    m = mixture.detach()[batches[0]:batches[1], layers[0]:layers[1]]
    c = constant.detach()[batches[0]:batches[1], layers[0]:layers[1]]
    n_batch = m.shape[0]
    n_layers = m.shape[1]
    xes = torch.cat((xv.reshape(-1, 1), yv.reshape(-1, 1)), 1).view(1, 1, -1, 2)
    rendering = (gm.evaluate(m, xes) + c.unsqueeze(-1)).view(n_batch, n_layers, width, height).transpose(2, 3)
    rendering = rendering.transpose(0, 1).reshape(n_layers * height, n_batch * width)
    return rendering


def render3d(mixture: Tensor, batches: index_range = (0, None), layers: index_range = (0, None),
             width: int = 100, height: int = 100, gm_vis_object: gm_vis.GMVisualizer = None):
    assert gm.n_dimensions(mixture) == 3
    assert gm.is_valid_mixture(mixture)

    end_gm_vis_object = False
    if gm_vis_object is None:
        gm_vis_object = gm_vis.GMVisualizer(False, width, height)
        gm_vis_object.set_camera_auto(True)
        gm_vis_object.set_density_rendering(True)
        end_gm_vis_object = True

    layer_start = layers[0]
    if layer_start is None:
        layer_start = 0
    layer_end = layers[1]
    if layer_end is None:
        layer_end = gm.n_layers(mixture)

    rendering_list = list()
    for lid in range(layer_start, layer_end):
        m = mixture[batches[0]:batches[1], lid:(lid+1), :, :]
        gm_vis_object.set_gaussian_mixtures(m.detach().cpu())
        rendering_list.append(torch.from_numpy(gm_vis_object.render()))

    rendering_tensor = torch.cat(rendering_list, dim=1)
    n_b = rendering_tensor.shape[0]
    height = rendering_tensor.shape[2]
    width = rendering_tensor.shape[3]
    n_l = len(rendering_list)

    if end_gm_vis_object:
        gm_vis_object.finish()

    return rendering_tensor.transpose(0, 1).reshape(n_l * height, n_b * width, 4)


def render_with_relu(mixture: Tensor, constant: Tensor,
                     batches: index_range = (0, None), layers: index_range = (0, None),
                     x_low: float = -22, y_low: float = -22, x_high: float = 22, y_high: float = 22,
                     width: int = 100, height: int = 100) -> Tensor:
    assert gm.is_valid_mixture_and_constant(mixture, constant)
    rendering = render(mixture, constant, batches, layers, x_low, y_low, x_high, y_high, width, height)
    return torch.max(rendering, torch.tensor([0.00001], dtype=torch.float32, device=mixture.device))
