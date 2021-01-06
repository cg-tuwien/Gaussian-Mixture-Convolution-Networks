from typing import Tuple

import torch
import gmc.mixture as gm
from enum import Enum


class ScalingMethod(Enum):
    SMALLEST_SIDE_TO_MAX = 1
    LARGEST_SIDE_TO_MAX = 2


class Scaler:
    # Tool for scaling Pointclouds and GMs to a given scale.
    # An instance of this is given a point cloud batch, to extract the required scaling from it.
    # The scaling will scale down these point clouds to a defined interval.
    # These extracted scalings can then be used to scale pointclouds and GMs.

    def __init__(self, active: bool = True,
                 interval: Tuple[float, float] = (0.0, 1.0),
                 scaling_method: ScalingMethod = ScalingMethod.SMALLEST_SIDE_TO_MAX):
        # Creates a new Scaler. For scaling to work, "set_pointcloud_batch" has to be called as well.
        # Parameters:
        #   active: bool
        #       Whether this scaler should actually perform scaling or not
        #   interval: Tuple[float, float]
        #       The interval the coordinates will be mapped to
        #   scaling_method: ScalingMethod
        #       ScalingMethod. Both methods move the minimum coordinates of a pointcloud to [min,min,min].
        #       If SMALLEST_SIDE_TO_MAX is chosen, the smallest side of the pc's bounding box will be mapped
        #       to the length max-min (so there might be coordinates with higher values than max), with
        #       LARGEST_SIDE_TO_MAX, the largest side will be mapped to that length.
        self._scaleP = self._scaleA = self._scaleC = self._offsetP = self._scaleL = None
        self._active = active
        self._min = interval[0]
        self._max = interval[1]
        self._scalingMethod = scaling_method

    def set_pointcloud_batch(self, pcbatch: torch.Tensor):
        # Has to be called before scaling!
        # This extracts the scalings from the pointcloud batch.
        # pcbatch is of size (batch_size, n_points, 3)

        batch_size = pcbatch.shape[0]

        if self._active:
            bbmin = torch.min(pcbatch, dim=1)[0]  # shape: (m, 3)
            bbmax = torch.max(pcbatch, dim=1)[0]  # shape: (m, 3)
            extends = bbmax - bbmin  # shape: (m, 3)

            # Scale point clouds to [0,1] in the smallest dimension
            if self._scalingMethod == ScalingMethod.SMALLEST_SIDE_TO_MAX:
                extends[extends.eq(0)] = float("Inf")
                self._scaleP = torch.min(extends, dim=1)[0]/(self._max - self._min)  # shape: (m)
            else:
                self._scaleP = torch.max(extends, dim=1)[0]/(self._max - self._min)  # shape: (m)
            self._offsetP = (bbmin / self._scaleP.unsqueeze(1) - self._min).view(-1, 1, 3)
            self._scaleP = self._scaleP.view(-1, 1, 1)  # shape: (m,1,1)
        else:
            self._scaleP = torch.ones(batch_size, 1, 1, dtype=pcbatch.dtype, device='cuda')  # shape: (m,1,1)
            self._offsetP = torch.zeros(batch_size, 1, 3, dtype=pcbatch.dtype, device='cuda')

        self._scaleA = torch.pow(self._scaleP, 3)  # shape: (m,1,1)
        self._scaleC = (self._scaleP ** 2).view(-1, 1, 1, 1, 1)  # shape: (m,1,1)
        self._scaleL = (3 * torch.log(self._scaleP)).view(-1)

    def scale_pc(self, pcbatch: torch.Tensor) -> torch.Tensor:
        # Scales the given point clouds to the scales extracted in set_pointcloud_batch
        # pcbatch is of size (batch_size, n_points, 3).
        # The scaled pointclouds are returned.
        if len(self._scaleP.shape) != 3:
            self._scaleP = self._scaleP.view(-1, 1, 1)
            self._offsetP = self._offsetP.view(-1, 1, 3)
        scaleddown = (pcbatch / self._scaleP) - self._offsetP
        return scaleddown

    def unscale_pc(self, pcbatch: torch.Tensor) -> torch.Tensor:
        # Scales the given point clouds to the scales extracted in set_pointcloud_batch
        # pcbatch is of size (batch_size, n_points, 3).
        # The scaled pointclouds are returned.
        if len(self._scaleP.shape) != 3:
            self._scaleP = self._scaleP.view(-1, 1, 1)
            self._offsetP = self._offsetP.view(-1, 1, 3)
        scaledup = (pcbatch + self._offsetP) * self._scaleP
        return scaledup

    def scale_gm(self, gmbatch: torch.Tensor) -> torch.Tensor:
        # Scales the given GMs (with amplitudes as weights) to the scales extracted in set_pointcloud_batch.
        # The scaled GMs are returned.
        if len(self._scaleP.shape) != 4:
            self._scaleP = self._scaleP.view(-1, 1, 1, 1)
            self._offsetP = self._offsetP.view(-1, 1, 1, 3)
        positions = gm.positions(gmbatch).clone()
        positions /= self._scaleP
        positions -= self._offsetP
        covariances = gm.covariances(gmbatch).clone()
        amplitudes = gm.weights(gmbatch).clone()
        amplitudes *= self._scaleA
        covariances /= self._scaleC
        return gm.pack_mixture(amplitudes, positions, covariances)

    def scale_gmm(self, gmmbatch: torch.Tensor) -> torch.Tensor:
        # Scales the given GMMs (with priors as weights!) from the scales extracted in set_pointcloud_batch to the
        # original scale.
        # The scaled GMs are returned.
        if len(self._scaleP.shape) != 4:
            self._scaleP = self._scaleP.view(-1, 1, 1, 1)
            self._offsetP = self._offsetP.view(-1, 1, 1, 3)
        positions = gm.positions(gmmbatch).clone()
        positions /= self._scaleP
        positions -= self._offsetP
        covariances = gm.covariances(gmmbatch).clone()
        covariances /= self._scaleC
        return gm.pack_mixture(gm.weights(gmmbatch).clone(), positions, covariances)

    def unscale_gm(self, gmbatch: torch.Tensor) -> torch.Tensor:
        # Scales the given GMs (with amplitudes as weights!) from the scales extracted in set_pointcloud_batch to the
        # original scale.
        # The scaled GMs are returned.
        if len(self._scaleP.shape) != 4:
            self._scaleP = self._scaleP.view(-1, 1, 1, 1)
            self._offsetP = self._offsetP.view(-1, 1, 1, 3)
        positions = gm.positions(gmbatch).clone()
        positions += self._offsetP
        positions *= self._scaleP
        covariances = gm.covariances(gmbatch).clone()
        amplitudes = gm.weights(gmbatch).clone()
        amplitudes /= self._scaleA
        covariances *= self._scaleC
        return gm.pack_mixture(amplitudes, positions, covariances)

    def unscale_gmm(self, gmmbatch: torch.Tensor) -> torch.Tensor:
        # Scales the given GMMs (with priors as weights!) from the scales extracted in set_pointcloud_batch to
        # the original scales.
        # The scaled GMs are returned.
        if len(self._scaleP.shape) != 4:
            self._scaleP = self._scaleP.view(-1, 1, 1, 1)
            self._offsetP = self._offsetP.view(-1, 1, 1, 3)
        positions = gm.positions(gmmbatch).clone()
        positions += self._offsetP
        positions *= self._scaleP
        covariances = gm.covariances(gmmbatch).clone()
        covariances *= self._scaleC
        return gm.pack_mixture(gm.weights(gmmbatch).clone(), positions, covariances)

    def unscale_gmm_wpc(self, weights: torch.Tensor, positions: torch.Tensor, covariances: torch.Tensor) \
            -> (torch.Tensor, torch.Tensor, torch.Tensor):
        # Scales the given GMMs (with priors as weights!) from the scales extracted in set_pointcloud_batch to
        # the original scales.
        # The scaled GMs are returned.
        if len(self._scaleP.shape) != 4:
            self._scaleP = self._scaleP.view(-1, 1, 1, 1)
            self._offsetP = self._offsetP.view(-1, 1, 1, 3)
        positions = positions.clone()
        positions += self._offsetP
        positions *= self._scaleP
        covariances = covariances.clone()
        covariances *= self._scaleC
        return weights.clone(), positions, covariances

    def unscale_losses(self, losses: torch.Tensor):
        # Scales negative Log-Likelihoods per Point from the extracted scales to the original scales.
        return losses + self._scaleL
