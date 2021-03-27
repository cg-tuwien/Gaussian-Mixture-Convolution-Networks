import math
from typing import List

import trimesh

from pcfitting import EvalFunction
import torch
import gmc.mixture as gm
import gmc.mat_tools as mat_tools
import numpy as np
import time

class AvgDensities(EvalFunction):
    # Calculates the average log likelihood of the point cloud given the mixture

    def __init__(self,
                 calculate_logavg: bool = True,
                 calculate_logstdv: bool = False,
                 calculate_logavg_scaled: bool = True,
                 calculate_avg: bool = True,
                 calculate_stdev: bool = True,
                 calculate_avg_scaled: bool = True,
                 calculate_stdev_scaled: bool = True,
                 calculate_cv: bool = True,
                 enlarge_evs: bool = False,
                 smallest_ev: float = 2e-4):
        self._logavg = calculate_logavg
        self._logstdev = calculate_logstdv
        self._logavg_scaled = calculate_logavg_scaled
        self._avg = calculate_avg
        self._stdev = calculate_stdev
        self._avg_scaled = calculate_avg_scaled
        self._stdev_scaled = calculate_stdev_scaled
        self._cv = calculate_cv
        self._n = self._logavg + self._logstdev + self._logavg_scaled + self._avg + self._stdev + self._avg_scaled + \
                  self._stdev_scaled + self._cv
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
            weights = gmamplitudes * (gmcovariances.det().sqrt() * 15.74960995)
            evals, evecs = torch.symeig(gmcovariances, eigenvectors=True)
            evals[evals.lt(self._smallest_ev)] = self._smallest_ev
            gmcovariances = evecs @ torch.diag_embed(evals) @ evecs.transpose(-1, -2)
            gminvcovariances = mat_tools.inverse(gmcovariances).contiguous()
            gmamplitudes = weights / (gmcovariances.det().sqrt() * 15.74960995)
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
        np.savetxt("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/EvalLogs/densities-" + str(round(time.time() * 1000)) + ".txt", output.cpu().numpy(),
                   delimiter="\n")
        res = torch.zeros(self._n, batch_size, device=pcbatch.device, dtype=pcbatch.dtype)
        (lgstd, lgmean) = torch.std_mean(torch.log(output), dim=1)
        (std, mean) = torch.std_mean(output, dim=1)
        i = 0
        if self._logavg:
            res[i, :] = lgmean
            i += 1
        if self._logstdev:
            res[i, :] = lgstd
            i += 1
        if self._logavg_scaled:
            res[i, :] = lgmean + self.calculate_scale_factor_log(modelpath)
            i += 1
        if self._avg:
            res[i, :] = mean
            i += 1
        if self._stdev:
            res[i, :] = std
            i += 1
        sf = 1
        if self._avg_scaled or self._stdev_scaled:
            sf = self.calculate_scale_factor(modelpath)
        if self._avg_scaled:
            res[i, :] = mean * sf
            i += 1
        if self._stdev_scaled:
            res[i, :] = std * sf
            i += 1
        if self._cv:
            res[i, :] = std / mean
            i += 1
        return res

    def get_names(self) -> List[str]:
        nlst = []
        if self._logavg:
            nlst.append("Average Log Density" + ("(evcorrected)" if self._enlarge_evs else ""))
        if self._logstdev:
            nlst.append("Stdev of Log Density" + ("(evcorrected)" if self._enlarge_evs else ""))
        if self._logavg_scaled:
            nlst.append("Average Log Density Scaled" + ("(evcorrected)" if self._enlarge_evs else ""))
        if self._avg:
            nlst.append("Average Density" + ("(evcorrected)" if self._enlarge_evs else ""))
        if self._stdev:
            nlst.append("Stdev of Density" + ("(evcorrected)" if self._enlarge_evs else ""))
        if self._avg_scaled:
            nlst.append("Average Density Scaled"+ ("(evcorrected)" if self._enlarge_evs else ""))
        if self._stdev_scaled:
            nlst.append("Stdev of Density Scaled" + ("(evcorrected)" if self._enlarge_evs else ""))
        if self._cv:
            nlst.append("Coefficient of Variation" + ("(evcorrected)" if self._enlarge_evs else ""))
        return nlst

    def calculate_scale_factor(self, modelpath: str):
        mesh = trimesh.load_mesh(modelpath)
        return math.pow(math.sqrt(mesh.area) / 128, 3)

    def calculate_scale_factor_log(self, modelpath: str):
       mesh = trimesh.load_mesh(modelpath)
       return 1.5 * math.log( (mesh.area / 16384))
