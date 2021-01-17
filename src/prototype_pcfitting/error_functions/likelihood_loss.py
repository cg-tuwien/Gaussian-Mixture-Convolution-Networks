import math

from prototype_pcfitting import ErrorFunction
import torch
import gmc.mixture as gm


class LikelihoodLoss(ErrorFunction):
    # Calculates an error by calculating the likelihood of the point cloud given the mixture

    def __init__(self, avoidinf: bool, eps: float = 1e-5):
        # avoidinf to True adds epps to the gaussian values to avoid taking the logarithm of 0 which would
        # lead to infinite loss
        self._avoidinf = avoidinf
        self._eps = eps

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
                + noisecontribution.view(batch_size, 1)
        res = -torch.mean(torch.log(output + (self._eps if self._avoidinf else 0)), dim=1)
        return res
