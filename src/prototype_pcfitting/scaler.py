import torch
import gmc.mixture as gm

class Scaler:

    def __init__(self):
        self.scale = self.scale2 = torch.ones(0,0,3)

    def setPointcloudBatch(self, pcbatch):
        batch_size = pcbatch.shape[0]
        bbmin = torch.min(pcbatch, dim=1)[0]  # shape: (m, 3)
        bbmax = torch.max(pcbatch, dim=1)[0]  # shape: (m, 3)
        extends = bbmax - bbmin  # shape: (m, 3)

        # Scale point clouds to [0,1] in the smallest dimension
        self.scale = torch.min(extends, dim=1)[0]  # shape: (m)
        self.scale = self.scale.view(-1, 1, 1)  # shape: (m,1,1)
        self.scale2 = self.scale ** 2
        scale2 = self.scale2.view(-1, 1, 1, 1, 1)  # shape: (m,1,1,1,1)
        # target = target / scale
        # target += 0.5
        # scale = scale.view(batch_size, 1, 1, 1)  # shape: (m,1,1,1)

    def scaleDownPC(self, pcbatch) -> torch.Tensor:
        if len(self.scale.shape) != 3:
            self.scale = self.scale.view(-1, 1, 1)
        scaleddown = pcbatch / self.scale
        scaleddown += 0.5
        return scaleddown

    def scaleUpPC(self, pcbatch) -> torch.Tensor:
        if len(self.scale.shape) != 3:
            self.scale = self.scale.view(-1, 1, 1)
        scaledup = pcbatch - 0.5
        scaledup *= self.scale
        return scaledup

    def scaleDownGM(self, gmbatch) -> torch.Tensor:
        if len(self.scale.shape) != 4:
            self.scale = self.scale.view(-1, 1, 1, 1)
        positions = gm.positions(gmbatch)
        positions /= self.scale
        positions += 0.5
        covariances = gm.covariances(gmbatch)
        amplitudes = gm.weights(gmbatch)
        amplitudes *= torch.pow(self.scale2, -2/3)
        covariances /= self.scale2
        return gm.pack_mixture(amplitudes, positions, covariances)

    def scaleUpGM(self, gmbatch) -> torch.Tensor:
        if len(self.scale.shape) != 4:
            self.scale = self.scale.view(-1, 1, 1, 1)
        positions = gm.positions(gmbatch)
        positions -= 0.5
        positions *= self.scale
        covariances = gm.covariances(gmbatch)
        amplitudes = gm.weights(gmbatch)
        amplitudes *= torch.pow(self.scale2, 2/3)
        covariances *= self.scale2
        return gm.pack_mixture(amplitudes, positions, covariances)