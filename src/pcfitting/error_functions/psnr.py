import math

from pcfitting import ErrorFunction, GMSampler
import torch
import gmc.mixture as gm
from pcfitting.cpp.gmeval import pyeval


class PSNR(ErrorFunction):
    # Calculates a score by calculating the PSNR of the point cloud given the mixture

    def calculate_score(self, pcbatch: torch.Tensor, gmpositions: torch.Tensor, gmcovariances: torch.Tensor,
                        gminvcovariances: torch.Tensor, gmamplitudes: torch.Tensor,
                        noisecontribution: torch.Tensor = None) -> torch.Tensor:
        point_size = pcbatch.shape[1]
        gmm = gm.convert_amplitudes_to_priors(gm.pack_mixture(gmamplitudes, gmpositions, gmcovariances))
        sampled = GMSampler.sample(gmm, point_size)

        psnr = pyeval.eval_psnr(pcbatch.view(point_size, 3), sampled.view(point_size, 3))
        psnr = torch.tensor([psnr], device=pcbatch.device, dtype=pcbatch.dtype)
        return psnr
