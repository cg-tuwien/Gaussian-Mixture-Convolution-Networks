from typing import List
from enum import Enum

import torch
import gmc.cpp.gm_vis.pygmvis as pygmvis   # This needs to be done prettier


class GmVisColoringRenderMode(Enum):
    COLOR_UNIFORM = 1,
    COLOR_WEIGHT = 2,
    COLOR_AMPLITUDE = 3


class GmVisColorRangeMode(Enum):
    RANGE_MANUAL = 1,
    RANGE_MINMAX = 2,
    RANGE_MEDMED = 3


class GmVisDensityRenderMode(Enum):
    ADDITIVE_EXACT = 1,
    ADDITIVE_ACC_OCTREE = 2,
    ADDITIVE_ACC_PROJECTED = 3,
    ADDITIVE_SAMPLING_OCTREE = 4


class GMVisualizer:

    def __init__(self, asyncmode: bool, width: int, height: int):
        self._vis = pygmvis.create_visualizer(asyncmode, width, height)

    def set_image_size(self, width: int, height: int):
        self._vis.set_image_size(width, height)

    def set_camera_auto(self, mode: bool):
        self._vis.set_camera_auto(mode)

    def set_view_matrix(self, viewmat: List[float]):
        self._vis.set_view_matrix(viewmat)

    def set_ellipsoid_rendering(self, ellipsoids: bool, pointcloud: bool = True):
        self._vis.set_ellipsoid_rendering(ellipsoids, pointcloud)

    def set_ellipsoid_coloring(self, colormode: GmVisColoringRenderMode, rangemode: GmVisColorRangeMode,
                               vmin: float = 0, vmax: float = 0):
        self._vis.set_ellipsoid_coloring(colormode, rangemode, vmin, vmax)

    def set_positions_rendering(self, positions: bool, pointcloud: bool = True):
        self._vis.set_positions_rendering(positions, pointcloud)

    def set_positions_coloring(self, colormode: GmVisColoringRenderMode, rangemode: GmVisColorRangeMode,
                               vmin: float = 0, vmax: float = 0):
        self._vis.set_positions_coloring(colormode, rangemode, vmin, vmax)

    def set_density_rendering(self, density: bool = True,
                              rendermode: GmVisDensityRenderMode = GmVisDensityRenderMode.ADDITIVE_EXACT):
        self._vis.set_density_rendering(density, rendermode)

    def set_density_coloring(self, automatic: bool = True, autoperc: float = 0.9, vmin: float = 0.0, vmax: float = 0.5):
        self._vis.set_density_coloring(automatic, autoperc, vmin, vmax)

    def set_density_accthreshold(self, automatic: bool = True, threshold: float = 0.0001):
        self._vis.set_density_accthreshold(automatic, threshold)

    def set_pointclouds(self, pointclouds: torch.Tensor):
        self._vis.set_pointclouds(pointclouds)

    def set_pointclouds_from_paths(self, paths: List[str]):
        self._vis.set_pointclouds_from_paths(paths)

    def set_gaussian_mixtures(self, mixtures: torch.Tensor, isgmm: bool):
        self._vis.set_gaussian_mixtures(mixtures, isgmm)

    def set_gaussian_mixtures_from_paths(self, paths: List[str], isgmm: bool):
        self._vis.set_gaussian_mixtures_from_paths(paths, isgmm)

    def set_callback(self, callback):
        self._vis.set_callback(callback)

    def render(self, epoch):
        self._vis.render(epoch)

    def finish(self):
        self._vis.finish()

    def force_stop(self):
        self._vis.forceStop()
