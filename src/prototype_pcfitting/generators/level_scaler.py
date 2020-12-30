import torch


class LevelScaler:
    # This class helps to scale a GM-level in hierarchical EM (Eckart HP)

    def __init__(self):
        self.scaleP = self.offsetP = None   # Scaling and Offset Values for points per Parent
        self.scaleL = None  # Scaling values for loss per parent
        self._parent_count = 0  # Number of parents
        self._parent_per_point = None  # Parent per point

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

        self.scaleP = torch.zeros(parent_count, dtype=pcbatch.dtype, device='cuda')  # shape: (parent_count)
        self.offsetP = torch.zeros(parent_count, 3, dtype=pcbatch.dtype, device='cuda')  # (parent_count, 3)
        for i in range(parent_count):
            rel_point_mask: torch.Tensor = torch.eq(parent_per_point, i)
            pcount = rel_point_mask.sum()
            if pcount > 1:
                rel_points = pcbatch[rel_point_mask]
                # not possible to parallelize batches as in each batch the rel point count might be different
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

        self.scaleL = (-3 * torch.log(self.scaleP)).view(-1)
        self.scaleP = self.scaleP.view(parent_count, 1)

    def scale_down_pc(self, pcbatch: torch.Tensor) -> torch.Tensor:
        # Scales down the given point cloud (1,n,3) according to the scales extracted in set_pointcloud_batch
        # The scaled pointcloud is returned (1,n,3).
        scaleddown = (pcbatch[0, :, :] - self.offsetP[self._parent_per_point, :]) \
                     / self.scaleP[self._parent_per_point, :]
        return scaleddown.view(1, -1, 3)

    def scale_up_gmm_wpc(self, weights: torch.Tensor, positions: torch.Tensor, covariances: torch.Tensor) -> \
            (torch.Tensor, torch.Tensor, torch.Tensor):
        # Scales up the given GMMs (with priors as weights!) according to the scales extracted in set_pointcloud_batch
        # The scaled GMs are returned. Batch size must be one!
        j = int(weights.shape[2] / self._parent_count)
        scale_position = self.scaleP.unsqueeze(0).expand(j, self._parent_count, 1).transpose(-2, -3).reshape(-1, 1)
        offset_position = self.offsetP.unsqueeze(0).expand(j, self._parent_count, 3).transpose(-2, -3).reshape(-1, 3)
        positions = positions.clone()
        positions[0, 0, :, :] *= scale_position
        positions[0, 0, :, :] += offset_position
        scale_covariances = (scale_position ** 2).view(-1, 1, 1)
        covariances = covariances.clone()
        covariances[0, 0, :, :, :] *= scale_covariances
        return weights.clone(), positions, covariances

    def scale_up_losses(self, loss_per_parent: torch.Tensor) -> torch.Tensor:
        # Scales up losses to original scale.
        #   loss_per_parent: torch.Tensor of size (1, parent_count) or (parent_count)
        return loss_per_parent + self.scaleL
