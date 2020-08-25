import torch
import gmc.mixture as gm


class Scaler:
    # Tool for scaling Pointclouds and GMs up and down.
    # An instance of this is given a point cloud batch, to extract the required down-scaling from it.
    # The scaling would scale down these point clouds to [0,1].
    # These extracted scalings can then be used to scale pointclouds and GMs.

    def __init__(self):
        self.scale = self.scale2a = self.scale2c = None

    def set_pointcloud_batch(self, pcbatch: torch.Tensor):
        # Has to be called before scaling!
        # This extracts the scalings from the pointcloud batch
        bbmin = torch.min(pcbatch, dim=1)[0]  # shape: (m, 3)
        bbmax = torch.max(pcbatch, dim=1)[0]  # shape: (m, 3)
        extends = bbmax - bbmin  # shape: (m, 3)

        # Scale point clouds to [0,1] in the smallest dimension
        self.scale = torch.min(extends, dim=1)[0]  # shape: (m)
        self.scale = self.scale.view(-1, 1, 1)  # shape: (m,1,1)
        self.scale2a = self.scale ** 2          # shape: (m,1,1)
        self.scale2c = self.scale2a.view(-1, 1, 1, 1, 1)  # shape: (m,1,1,1,1)

    def scale_down_pc(self, pcbatch: torch.Tensor) -> torch.Tensor:
        # Scales down the given point clouds according to the scales extracted in set_pointcloud_batch
        # The scaled pointclouds are returned.
        if len(self.scale.shape) != 3:
            self.scale = self.scale.view(-1, 1, 1)
        scaleddown = pcbatch / self.scale
        scaleddown += 0.5
        return scaleddown

    def scale_up_pc(self, pcbatch: torch.Tensor) -> torch.Tensor:
        # Scales up the given point clouds according to the scales extracted in set_pointcloud_batch
        # The scaled pointclouds are returned.
        if len(self.scale.shape) != 3:
            self.scale = self.scale.view(-1, 1, 1)
        scaledup = pcbatch - 0.5
        scaledup *= self.scale
        return scaledup

    def scale_down_gm(self, gmbatch: torch.Tensor) -> torch.Tensor:
        # Scales down the given GMs according to the scales extracted in set_pointcloud_batch
        # The scaled GMs are returned.
        if len(self.scale.shape) != 4:
            self.scale = self.scale.view(-1, 1, 1, 1)
        positions = gm.positions(gmbatch)
        positions /= self.scale
        positions += 0.5
        covariances = gm.covariances(gmbatch)
        amplitudes = gm.weights(gmbatch)
        amplitudes *= torch.pow(self.scale2a, -2/3)
        covariances /= self.scale2c
        return gm.pack_mixture(amplitudes, positions, covariances)

    def scale_up_gm(self, gmbatch: torch.Tensor) -> torch.Tensor:
        # Scales up the given GMs according to the scales extracted in set_pointcloud_batch
        # The scaled GMs are returned.
        if len(self.scale.shape) != 4:
            self.scale = self.scale.view(-1, 1, 1, 1)
        positions = gm.positions(gmbatch)
        positions -= 0.5
        positions *= self.scale
        covariances = gm.covariances(gmbatch)
        amplitudes = gm.weights(gmbatch)
        amplitudes *= torch.pow(self.scale2a, 2/3)
        covariances *= self.scale2c
        return gm.pack_mixture(amplitudes, positions, covariances)
