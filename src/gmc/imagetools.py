import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt

import colourmap

gamma_correction = 1/2.2


def colour_mapped(mono, low, high):
    if mono.ndim > 2:
        raise Exception("colour_map is only applicable for mono matrices")

    normaliser = matplotlib.colors.Normalize(vmin=low, vmax=high, clip=True)
    mapper = matplotlib.cm.ScalarMappable(norm=normaliser, cmap=madam_cm.cm_linSeg)
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
