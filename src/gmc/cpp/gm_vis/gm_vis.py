from typing import List, Tuple
from enum import IntEnum
import numpy as np

import torch
import gmc.cpp.gm_vis.build.pygmvis as pygmvis


def onetime_density_render(width: int,
                         height: int,
                         mixtures: torch.Tensor,
                         logarithmic:bool = False) -> np.array:
    # Size of return: (batchsize, 1, height, width, 4)
    vis = GMVisualizer(False, width, height)
    vis.set_camera_auto(True)
    vis.set_density_rendering(True)
    vis.set_density_logarithmic(logarithmic)
    vis.set_gaussian_mixtures(mixtures.detach().cpu(), False)
    res = vis.render(0)
    vis.finish()
    return res

def onetime_ellipsoids_render(width: int,
                            height: int,
                            mixtures: torch.Tensor) -> torch.Tensor:
    # Size of return: (batchsize, 1, height, width, 4)
    vis = GMVisualizer(False, width, height)
    vis.set_camera_auto(True)
    vis.set_ellipsoids_rendering(True)
    vis.set_ellipsoids_colormode(GmVisColoringRenderMode.COLOR_WEIGHT)
    vis.set_ellipsoids_rangemode(GmVisColorRangeMode.RANGE_MINMAX)
    vis.set_gaussian_mixtures(mixtures.detach().cpu(), False)
    res = vis.render()
    vis.finish()
    return res

class GmVisColoringRenderMode(IntEnum):
    COLOR_UNIFORM = 1,
    COLOR_WEIGHT = 2,
    COLOR_AMPLITUDE = 3

    def convert_to_cpp(self):
        if self == GmVisColoringRenderMode.COLOR_UNIFORM:
            return pygmvis.GMColoringRenderMode.COLOR_UNIFORM
        elif self == GmVisColoringRenderMode.COLOR_WEIGHT:
            return pygmvis.GMColoringRenderMode.COLOR_WEIGHT
        elif self == GmVisColoringRenderMode.COLOR_AMPLITUDE:
            return pygmvis.GMColoringRenderMode.COLOR_AMPLITUDE
        return None


class GmVisColorRangeMode(IntEnum):
    RANGE_MANUAL = 1,
    RANGE_MINMAX = 2,
    RANGE_MEDMED = 3

    def convert_to_cpp(self):
        if self == GmVisColorRangeMode.RANGE_MANUAL:
            return pygmvis.GMColorRangeMode.RANGE_MANUAL
        elif self == GmVisColorRangeMode.RANGE_MINMAX:
            return pygmvis.GMColorRangeMode.RANGE_MINMAX
        elif self == GmVisColorRangeMode.RANGE_MEDMED:
            return pygmvis.GMColorRangeMode.RANGE_MEDMED
        return None


class GmVisDensityRenderMode(IntEnum):
    ADDITIVE_EXACT = 1,
    ADDITIVE_ACC_OCTREE = 2,
    ADDITIVE_ACC_PROJECTED = 3,
    ADDITIVE_SAMPLING_OCTREE = 4

    def convert_to_cpp(self):
        if self == GmVisDensityRenderMode.ADDITIVE_EXACT:
            return pygmvis.GMDensityRenderMode.ADDITIVE_EXACT
        elif self == GmVisDensityRenderMode.ADDITIVE_ACC_OCTREE:
            return pygmvis.GMDensityRenderMode.ADDITIVE_ACC_OCTREE
        elif self == GmVisDensityRenderMode.ADDITIVE_ACC_PROJECTED:
            return pygmvis.GMDensityRenderMode.ADDITIVE_ACC_PROJECTED
        elif self == GmVisDensityRenderMode.ADDITIVE_SAMPLING_OCTREE:
            return pygmvis.GMDensityRenderMode.ADDITIVE_SAMPLING_OCTREE
        return None


class GMVisualizer:

    def __init__(self, asyncmode: bool, width: int, height: int):
        # Creates the visualizer (only one visualizer should be active at a time)
        # Parameters:
        #   asyncmode: bool
        #       If true, the renderings will rone in a new thread. If false, we'll wait until it's done
        #   width/height: int
        #       Size of desired renderings in pixels
        self._vis = pygmvis.create_visualizer(asyncmode, width, height)

    def set_image_size(self, width: int, height: int):
        # Sets the size of the renderings
        self._vis.set_image_size(width, height)

    def set_camera_auto(self, mode: bool):
        # If mode is true, the camera position will be set automatically
        # according to the pointcloud if given, otherwise according to the gmm
        self._vis.set_camera_auto(mode)

    def set_camera_lookat(self, positions: Tuple[float, float, float], lookat: Tuple[float, float, float],
                          up: Tuple[float, float, float]):
        # Sets the camera position given three 3d-vectors in lookat-style
        self._vis.set_camera_lookat(positions, lookat, up)

    def set_view_matrix(self, viewmat: List[float]):
        # Sets the view matrix
        self._vis.set_view_matrix(viewmat)

    def set_ellipsoids_rendering(self, ellipsoids: bool, pointcloud: bool = True):
        # Activates or disables Ellipsoid Rendering
        # Parameters:
        #   ellipsoids: bool
        #       If ellipsoids should be rendered or not
        #   pointcloud: bool
        #       If the pointcloud should be visible in the ellipsoids rendering
        self._vis.set_ellipsoids_rendering(ellipsoids, pointcloud)

    def set_ellipsoids_colormode(self, colormode: GmVisColoringRenderMode):
        # Sets the color mode of ellipsoids (Uniform, by weight, by amplitude)
        self._vis.set_ellipsoids_colormode(colormode.convert_to_cpp())

    def set_ellipsoids_rangemode(self, rangemode: GmVisColorRangeMode, vmin: float = 1, vmax: float = 0):
        # If color mode is not uniform, this sets the range mode for coloring: minmax, medmed or manual.
        # If manual, the manual min and max values can be set
        self._vis.set_ellipsoids_rangemode(rangemode.convert_to_cpp(), vmin, vmax)

    def set_positions_rendering(self, positions: bool, pointcloud: bool = True):
        # Activates or disables GM Positions Rendering
        # Parameters:
        #   positions: bool
        #       If positions should be rendered or not
        #   pointcloud: bool
        #       If the pointcloud should be visible in the ellipsoids rendering
        self._vis.set_positions_rendering(positions, pointcloud)

    def set_positions_colormode(self, colormode: GmVisColoringRenderMode):
        # Sets the color mode of positions (Uniform, by weight, by amplitude)
        self._vis.set_positions_colormode(colormode.convert_to_cpp())

    def set_positions_rangemode(self, rangemode: GmVisColorRangeMode, vmin: float = 0, vmax: float = 0):
        # If color mode is not uniform, this sets the range mode for coloring: minmax, medmed or manual.
        # If manual, the manual min and max values can be set
        self._vis.set_positions_rangemode(rangemode.convert_to_cpp(), vmin, vmax)

    def set_density_rendering(self, density: bool):
        # Activates or disables density rendering
        self._vis.set_density_rendering(density)

    def set_density_rendermode(self, rendermode: GmVisDensityRenderMode):
        # Sets the density render mode (exact, octree, projected, sampling) (Default: Exact)
        self._vis.set_density_rendermode(rendermode.convert_to_cpp())

    def set_density_range_auto(self, autoperc: float = 0.75):
        # Enables automatic detection of density color range according to bounding box size (activated by default).
        # Additionally, a value autoperc between 0 and 1 can be set to change the intensity (default: 0.75),
        # corresponds to the slider in the UI
        self._vis.set_density_range_auto(autoperc)

    def set_density_range_manual(self, min: float, max: float):
        # Disables automatic detection of density color range and sets it manually given the min and max values.
        self._vis.set_density_range_manual(min, max)

    def set_density_logarithmic(self, logarithmic: bool):
        # Enables or disables logarithmic mode (Default: False)
        self._vis.set_density_logarithmic(logarithmic)

    def set_density_accthreshold(self, automatic: bool = True, threshold: float = 0.0001):
        # Sets the acceleration threshold for density rendermode ADDITIVE_ACC_PROJECTED
        # automatic determines if the threshold should be detected automatically (default)
        # if false, threshold defines the threshold
        self._vis.set_density_accthreshold(automatic, threshold)

    def set_pointclouds(self, pointclouds: torch.Tensor):
        # Sets the pointclouds
        self._vis.set_pointclouds(pointclouds)

    def set_pointclouds_from_paths(self, paths: List[str]):
        # Sets the pointclouds as file paths
        self._vis.set_pointclouds_from_paths(paths)

    def set_gaussian_mixtures(self, mixtures: torch.Tensor, isgmm: bool = False):
        # Sets the gaussian mixtures.
        # mixtures: tensor of size (bs, 1, ng, 13)
        # isgmm: If true, the weights of the mixture are prior-weights of a gmm, if false they are amplitudes
        self._vis.set_gaussian_mixtures(mixtures, isgmm)

    def set_gaussian_mixtures_from_paths(self, paths: List[str], isgmm: bool = False):
        # Sets the gaussian mixtures as file paths.
        # mixtures: tensor of size (bs, 1, ng, 13)
        # isgmm: If true, the weights of the mixture are prior-weights of a gmm, if false they are amplitudes
        self._vis.set_gaussian_mixtures_from_paths(paths, isgmm)

    def set_callback(self, callback):
        # Sets a render callback, that is called in async mode when rendering is finished
        # The callback has the following parameters:
        # (epoch: int, pixeldata: np.array/torch.tensor, gmidx: int, ridx: int)
        #   epoch: int
        #       The value of epoch given at the render call (for identification)
        #   pixeldata: np.array
        #       The pixeldata of the image
        #   gmidx: int
        #       The index of the corresponding GMM in the batch
        #   ridx: int
        #       Identifies which rendering this is. 0 = Ellipsoids, 1 = Positions, 2 = Density
        self._vis.set_callback(callback)

    def render(self, epoch=0):
        # Renders the previously given mixture with the set options
        # If asyncmode is disabled, this returns a np.array of size (bs, ri, height, width, 4)
        # where bs is the batch size and ri the amount of enabled renderings (in order ellipsoids, positions, density).
        # Otherwise the result is given through a callback.
        # Epoch is an optional value that is given to the callback.
        return self._vis.render(epoch)

    def finish(self):
        # Has to be called when done using the visualizer!
        self._vis.finish()

    def force_stop(self):
        self._vis.forceStop()