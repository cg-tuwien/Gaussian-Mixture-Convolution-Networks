import math
from typing import List

import trimesh

from pcfitting import EvalFunction
import torch
import gmc.mixture as gm
import gmc.mat_tools as mat_tools


class AvgLogLikelihood(EvalFunction):
    # Calculates the average log likelihood of the point cloud given the mixture

    def __init__(self,
                 calculate_stdev: bool = True,
                 calculate_rel_to_gt: bool = True,
                 enlarge_evs: bool = True,
                 smallest_ev: float = 2e-4):
        self._stdev = calculate_stdev
        self._rel_to_gt = calculate_rel_to_gt
        self._enlarge_evs = enlarge_evs
        self._smallest_ev = smallest_ev
        pass

    def calculate_score(self, pcbatch: torch.Tensor, gmpositions: torch.Tensor, gmcovariances: torch.Tensor,
                        gminvcovariances: torch.Tensor, gmamplitudes: torch.Tensor,
                        noisecontribution: torch.Tensor = None, modelpath: str = None) -> torch.Tensor:
        batch_size = pcbatch.shape[0]
        points = pcbatch.view(batch_size, 1, -1, 3)
        point_count = points.shape[2]
        if self._enlarge_evs:
            evals, evecs = torch.symeig(gmcovariances, eigenvectors=True)
            evals[evals.lt(self._smallest_ev)] = self._smallest_ev
            gmcovariances = evecs @ torch.diag_embed(evals) @ evecs.transpose(-1, -2)
            gminvcovariances = mat_tools.inverse(gmcovariances).contiguous()
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
        res = torch.zeros(1 + self._stdev + self._rel_to_gt, batch_size, device=pcbatch.device, dtype=pcbatch.dtype)
        if not self._stdev:
            res[0, :] = torch.mean(torch.log(output), dim=1)
            if self._rel_to_gt:
                res[1, :] = res[0, :] - self.calculate_ground_truth(modelpath)
            return res
        else:
            (std, mean) = torch.std_mean(torch.log(output), dim=1)
            res[0, :] = mean
            res[1, :] = std
            if self._rel_to_gt:
                res[2, :] = mean - self.calculate_ground_truth(modelpath)
            return res

    def get_names(self) -> List[str]:
        nlst = ["Average Log Likelihood" + ("(evcorrected)" if self._enlarge_evs else "")]
        if self._stdev:
            nlst.append("Stdev of Log Likelihood" + ("(evcorrected)" if self._enlarge_evs else ""))
        if self._rel_to_gt:
            nlst.append("Average Log Likelihood RelGT" + ("(evcorrected)" if self._enlarge_evs else ""))
        return nlst

    def calculate_ground_truth(self, modelpath: str):
       mesh = trimesh.load_mesh(modelpath)
       return math.log(1 / self._smallest_ev * mesh.area)
