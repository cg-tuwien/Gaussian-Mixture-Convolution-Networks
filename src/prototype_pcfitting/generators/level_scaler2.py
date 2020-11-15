import torch
import gmc.mixture as gm
from enum import Enum
import numpy as np

# ToDo: Clean up, document!

class LevelScaler2:

    def __init__(self):
        self.scaleP = None

    def set_pointcloud(self, pcbatch: torch.Tensor, parent_per_point: torch.Tensor, parent_count: int):
        # Has to be called before scaling!
        # This extracts the scalings from the pointcloud (1, n, 3)
        # parent_per_point: (1, n)
        n = parent_per_point.shape[1]
        m = parent_count

        self._m = m
        self._parent_per_point = parent_per_point

        self.scaleP = torch.zeros(m).to(pcbatch.dtype).cuda() # shape: (m)
        self.offsetP = torch.zeros(m, 3).to(pcbatch.dtype).cuda() # (m,3)
        for i in range(m):
            rel_point_mask = (parent_per_point == i)
            pcount = rel_point_mask.sum()
            if pcount > 1:
                rel_points = pcbatch[rel_point_mask]
                # this is not possible to parallelize among different batches as in each batch the rel point count might be different
                rel_bbmin = torch.min(rel_points, dim=0)[0]
                rel_bbmax = torch.max(rel_points, dim=0)[0]
                rel_extends = rel_bbmax - rel_bbmin
                self.scaleP[i] = torch.max(rel_extends)
                self.offsetP[i] = rel_bbmin
            elif pcount == 1:
                self.scaleP[i] = torch.tensor(1.0)
                self.offsetP[i] = pcbatch[rel_point_mask].view(3)
            else:
                self.scaleP[i] = torch.tensor(1.0)
                self.offsetP[i] = torch.tensor([0.0, 0.0, 0.0])

        self.scaleP = self.scaleP.view(m, 1)

    def scale_down_pc(self, pcbatch: torch.Tensor) -> torch.Tensor:
        # Scales down the given point clouds according to the scales extracted in set_pointcloud_batch
        # The scaled pointclouds are returned. pcbatch (1,n,3)
        # was wir eigentlich meinen ist self.scaleP[relevant_parents.indizesof(parent_per_point[0,:])]
        # sorter = np.argsort(self._relevant_parents[0, :])
        # indizes = sorter[np.searchsorted(self._relevant_parents[0, :], self._parent_per_point[0, :], sorter=sorter)]
        scaleddown = (pcbatch[0,:,:] - self.offsetP[self._parent_per_point, :]) / self.scaleP[self._parent_per_point, :]
        return scaleddown.view(1, -1, 3)

    def scale_up_gmm_wpc(self, weights: torch.Tensor, positions: torch.Tensor, covariances: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        # Scales up the given GMMs (with priors as weights!) according to the scales extracted in set_pointcloud_batch
        # The scaled GMs are returned. BATCH SIZE MUST BE ONE
        j = int(weights.shape[2] / self._m)
        # repeat scale j times for each sub gmm
        scalePosition = self.scaleP.unsqueeze(0).expand(j, self._m, 1).transpose(-2, -3).reshape(-1, 1)
        offsetPosition = self.offsetP.unsqueeze(0).expand(j, self._m, 3).transpose(-2, -3).reshape(-1, 3)
        positions = positions.clone()
        positions[0, 0, :, :] *= scalePosition
        positions[0, 0, :, :] += offsetPosition
        scaleCovariances = (scalePosition ** 2).view(-1, 1, 1)
        covariances = covariances.clone()
        covariances[0, 0, :, :, :] *= scaleCovariances
        return weights.clone(), positions, covariances
