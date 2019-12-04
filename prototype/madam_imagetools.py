import numpy as np
import matplotlib.cm

import madam_cm

gamma_correction = 1/2.2


def colour_mapped(mono, low, high):
    if mono.ndim > 2:
        raise Exception("colour_map is only applicable for mono matrices")

    normaliser = matplotlib.colors.Normalize(vmin=low, vmax=high, clip=True)
    mapper = matplotlib.cm.ScalarMappable(norm=normaliser, cmap=madam_cm.cm_linSeg)
    return mapper.to_rgba(mono)