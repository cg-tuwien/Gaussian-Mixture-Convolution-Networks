import math
from typing import List

from pcfitting import EvalFunction, GMSampler, data_loading
import torch
import gmc.mixture as gm
import pcfitting.config as general_config
from pytorch3d import loss as p3dl


class RcdLoss(EvalFunction):
    # Calculates an error by calculating the reconstruction error of a mixture.
    # This ignores prior weights! The weights are assumed to be equal! To create usable gradients, one Gaussian
    # is sampled in the beginning and then the point cloud is just scaled and moved to fit the constructed Gaussians.
    # This is used by GradientDescentRecGenerator

    def __init__(self, np, ng):
        # avoidinf to True adds epps to the gaussian values to avoid taking the logarithm of 0 which would
        # lead to infinite loss
        g = gm.pack_mixture(torch.tensor([1]).view(1,1,-1), torch.tensor([0,0,0]).view(1,1,-1,3), torch.eye(3).view(1,1,-1,3, 3)).cuda()
        self._pointsperG = int(np / ng)
        self._template = GMSampler.sampleGMM_ext(g, self._pointsperG).view(1, 1, self._pointsperG, 3).expand(1, ng, self._pointsperG, 3)

    def calculate_score(self, pcbatch: torch.Tensor, gmpositions: torch.Tensor, gmcovariances: torch.Tensor,
                        gminvcovariances: torch.Tensor, gmamplitudes: torch.Tensor,
                        noisecontribution: torch.Tensor = None, modelpath: str = None) -> torch.Tensor:
        batch_size = pcbatch.shape[0]
        points = pcbatch.view(batch_size, 1, -1, 3)
        point_count = points.shape[2]
        gauss_count = gmpositions.shape[2]

        eigenvalues, eigenvectors = torch.symeig(gmcovariances[:, 0, :, :, :], eigenvectors=True)
        trans = torch.sqrt(eigenvalues).unsqueeze(-1).expand(1,gauss_count,3,3) * eigenvectors.transpose(-1,-2)
        pointscaled = torch.matmul(self._template.unsqueeze(-2), trans.unsqueeze(2).expand(batch_size, gauss_count, self._pointsperG, 3, 3)).squeeze(-2)
        pointsmoved = pointscaled + gmpositions.view(batch_size, gauss_count, 1, 3).expand(batch_size, gauss_count, self._pointsperG, 3)

        # data_loading.write_pc_to_off(r"C:\Users\SimonFraiss\Desktop\rcd.off", pointsmoved.view(batch_size, -1, 3))

        loss = torch.zeros(1, batch_size).cuda()
        for i in range(batch_size):
            loss[0, i], _ = p3dl.chamfer_distance(pcbatch[i].view(1,-1,3), pointsmoved[i].view(1,-1, 3))
        return loss.sqrt()

    def get_names(self) -> List[str]:
        return ["RcdLoss"]