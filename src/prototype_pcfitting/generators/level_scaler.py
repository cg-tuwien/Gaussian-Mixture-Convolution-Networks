from typing import Tuple

import torch


class LevelScaler:
    # This class helps to scale a GM-level in hierarchical EM (Eckart HP)
    # Each Sub-GM is scaled differently.
    # Currently this is not used (active is always false, leading the scaler do not perform any operation)

    def __init__(self, active: bool = True, interval: Tuple[float, float] = (0.0, 1.0)):
        self._scaleP = self._offsetP = None   # Scaling and Offset Values for points per Parent
        self._parent_count = 0  # Number of parents
        self._parent_per_point = None  # Parent per point
        self._active = active
        self._min = interval[0]
        self._max = interval[1]

    def set_pointcloud(self, pcbatch: torch.Tensor, parent_per_point: torch.Tensor, parent_count: int):
        # Has to be called before scaling!
        # This extracts the scalings from the pointcloud. Each subpointcloud of points belonging to the same parent
        # is downscaled to [0,1].
        # pcbatch: torch.Tensor of size (1, n, 3)
        #   reference pointcloud
        # parent_per_point: torch.Tensor of size (1, n)
        #   Parent ID for each point. Points with the same parent are grouped together and get the same scaling.
        # parent_count: int
        #   Number of relevant parents

        self._parent_count = parent_count
        self._parent_per_point = parent_per_point

        if self._active:
            self._scaleP = torch.zeros(parent_count, dtype=pcbatch.dtype, device='cuda')  # shape: (parent_count)
            self._offsetP = torch.zeros(parent_count, 3, dtype=pcbatch.dtype, device='cuda')  # (parent_count, 3)
            for i in range(parent_count):
                rel_point_mask: torch.Tensor = torch.eq(parent_per_point, i)
                pcount = rel_point_mask.sum()
                if pcount > 1:
                    rel_points = pcbatch[rel_point_mask]
                    # not possible to parallelize batches as in each batch the rel point count might be different
                    rel_bbmin = torch.min(rel_points, dim=0)[0]
                    rel_bbmax = torch.max(rel_points, dim=0)[0]
                    rel_extends = rel_bbmax - rel_bbmin
                    self._scaleP[i] = torch.max(rel_extends) / (self._max - self._min)
                    self._offsetP[i] = rel_bbmin / self._scaleP[i] - self._min
                elif pcount == 1:
                    self._scaleP[i] = torch.tensor(1.0)
                    self._offsetP[i] = pcbatch[rel_point_mask].view(3) - self._min
                else:
                    self._scaleP[i] = torch.tensor(1.0)
                    self._offsetP[i] = torch.tensor([0.0, 0.0, 0.0])
            self._scaleP = self._scaleP.view(parent_count, 1)
        else:
            self._scaleP = torch.ones(parent_count, 1, dtype=pcbatch.dtype, device='cuda')
            self._offsetP = torch.zeros(parent_count, 3, dtype=pcbatch.dtype, device='cuda')

    def scale_pc(self, pcbatch: torch.Tensor) -> torch.Tensor:
        # Scales down the given point cloud (1,n,3) according to the scales extracted in set_pointcloud_batch
        # The scaled pointcloud is returned (1,n,3).
        scaleddown = (pcbatch[0, :, :] / self._scaleP[self._parent_per_point, :]) \
                     - self._offsetP[self._parent_per_point, :]
        return scaleddown.view(1, -1, 3)

    def unscale_gmm_wpc(self, weights: torch.Tensor, positions: torch.Tensor, covariances: torch.Tensor) -> \
            (torch.Tensor, torch.Tensor, torch.Tensor):
        # Scales up the given GMMs (with priors as weights!) according to the scales extracted in set_pointcloud_batch
        # The scaled GMs are returned. Batch size must be one!
        j = int(weights.shape[2] / self._parent_count)
        scale_position = self._scaleP.unsqueeze(0).expand(j, self._parent_count, 1).transpose(-2, -3).reshape(-1, 1)
        offset_position = self._offsetP.unsqueeze(0).expand(j, self._parent_count, 3).transpose(-2, -3).reshape(-1, 3)
        positions = positions.clone()
        positions[0, 0, :, :] += offset_position
        positions[0, 0, :, :] *= scale_position
        scale_covariances = (scale_position ** 2).view(-1, 1, 1)
        covariances = covariances.clone()
        covariances[0, 0, :, :, :] *= scale_covariances
        return weights.clone(), positions, covariances
