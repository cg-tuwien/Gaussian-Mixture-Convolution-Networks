import math
from typing import List

from pcfitting import EvalFunction, GMSampler
import torch
import gmc.mixture as gm
from pcfitting.cpp.gmeval import pyeval


class PSNR(EvalFunction):
    # Calculates a score by calculating the PSNR of the point cloud given the mixture

    def calculate_score(self, pcbatch: torch.Tensor, gmpositions: torch.Tensor, gmcovariances: torch.Tensor,
                        gminvcovariances: torch.Tensor, gmamplitudes: torch.Tensor,
                        noisecontribution: torch.Tensor = None, modelpath: str = None) -> torch.Tensor:
        assert pcbatch.shape[0] == 1
        point_size = pcbatch.shape[1]
        gmm = gm.convert_amplitudes_to_priors(gm.pack_mixture(gmamplitudes, gmpositions, gmcovariances))
        sampled = GMSampler.sampleGMM(gmm, point_size)

        psnr = pyeval.eval_psnr(pcbatch.view(point_size, 3), sampled.view(point_size, 3))
        psnr = torch.tensor([psnr], device=pcbatch.device, dtype=pcbatch.dtype)
        return psnr.view(1, -1)

    def get_names(self) -> List[str]:
        return ["PSNR"]
