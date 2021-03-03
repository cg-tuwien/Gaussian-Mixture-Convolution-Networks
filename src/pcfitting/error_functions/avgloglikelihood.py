import math
from typing import List

from pcfitting import EvalFunction
import torch
import gmc.mixture as gm


class AvgLogLikelihood(EvalFunction):
    # Calculates the average log likelihood of the point cloud given the mixture

    def __init__(self,
                 calculate_stdev: bool = True):
        self._stdev = calculate_stdev
        pass

    def calculate_score(self, pcbatch: torch.Tensor, gmpositions: torch.Tensor, gmcovariances: torch.Tensor,
                        gminvcovariances: torch.Tensor, gmamplitudes: torch.Tensor,
                        noisecontribution: torch.Tensor = None) -> torch.Tensor:
        batch_size = pcbatch.shape[0]
        points = pcbatch.view(batch_size, 1, -1, 3)
        point_count = points.shape[2]
        mixture_with_inversed_cov = gm.pack_mixture(gmamplitudes, gmpositions, gminvcovariances)
        output = torch.zeros(batch_size, point_count, dtype=points.dtype, device='cuda')
        subbatches = math.ceil((batch_size * point_count) / 65535)
        subbatch_pointcount = math.ceil(point_count / subbatches)
        for p in range(subbatches):
            startidx = p * subbatch_pointcount
            endidx = min((p + 1) * subbatch_pointcount, point_count)
            output[:, startidx:endidx] = \
                gm.evaluate_inversed(mixture_with_inversed_cov, points[:, :, startidx:endidx, :]).view(batch_size, -1) \
                + (noisecontribution.view(batch_size, 1) if noisecontribution is not None else 0)
        if not self._stdev:
            mean = torch.mean(torch.log(output), dim=1)
            return mean.view(1, -1)
        else:
            (std, mean) = torch.std_mean(torch.log(output), dim=1)
            res = torch.zeros(2, batch_size, device=mean.device, dtype=mean.dtype)
            res[0, :] = mean
            res[1, :] = std
            return res

    def get_names(self) -> List[str]:
        nlst = ["Average Log Likelihood"]
        if self._stdev:
            nlst.append("Stdev of Log Likelihood")
        return nlst