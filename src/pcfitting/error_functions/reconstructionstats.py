import math
from typing import List

import trimesh
import trimesh.sample

from pcfitting import EvalFunction, GMSampler
import torch
import gmc.mixture as gm
import gmc.mat_tools as mat_tools
from pcfitting.cpp.gmeval import pyeval


class ReconstructionStats(EvalFunction):
    # Calculates the average log likelihood of the point cloud given the mixture

    def __init__(self,
                 rmsd_pure: bool = True,
                 rmsd_scaled_bb_diag: bool = True,
                 rmsd_scaled_by_area: bool = True,
                 md_pure: bool = True,
                 md_scaled_bb_diag: bool = True,
                 md_scaled_by_area: bool = True,
                 stdev: bool = True,
                 stdev_scaled_bb_diag: bool = True,
                 stdev_scaled_by_area: bool = True,
                 psnr: bool = True,
                 sample_points: int = 10000):
        self._rmsd_pure = rmsd_pure
        self._rmsd_scaled_bb_diag = rmsd_scaled_bb_diag
        self._rmsd_scaled_by_area = rmsd_scaled_by_area
        self._md_pure = md_pure
        self._md_scaled_bb_diag = md_scaled_bb_diag
        self._md_scaled_by_area = md_scaled_by_area
        self._stdev = stdev
        self._stdev_scaled_bb_diag = stdev_scaled_bb_diag
        self._stdev_scaled_by_area = stdev_scaled_by_area
        self._psnr = psnr
        self._samplepoints = sample_points
        pass

    def calculate_score(self, pcbatch: torch.Tensor, gmpositions: torch.Tensor, gmcovariances: torch.Tensor,
                        gminvcovariances: torch.Tensor, gmamplitudes: torch.Tensor,
                        noisecontribution: torch.Tensor = None, modelpath: str = None) -> torch.Tensor:
        batch_size = pcbatch.shape[0]
        assert batch_size == 1

        nmeth = self._rmsd_pure + self._rmsd_scaled_bb_diag + self._rmsd_scaled_by_area + self._md_pure \
                + self._md_scaled_bb_diag + self._md_scaled_by_area + self._stdev \
                + self._stdev_scaled_bb_diag + self._stdev_scaled_by_area + self._psnr
        result = torch.zeros(nmeth, batch_size, device=pcbatch.device, dtype=pcbatch.dtype)

        gmm = gm.convert_amplitudes_to_priors(gm.pack_mixture(gmamplitudes, gmpositions, gmcovariances))
        sampled = GMSampler.sample(gmm, self._samplepoints)

        pmin = torch.min(pcbatch[0], dim=0)[0]
        pmax = torch.max(pcbatch[0], dim=0)[0]
        bbscale = torch.norm(pmax - pmin).item()

        i = 0
        rmsd, md, stdev = pyeval.eval_rmsd_unscaled(pcbatch.view(-1, 3), sampled.view(-1, 3))
        scalefactor = 1
        if self._stdev_scaled_by_area or self._md_scaled_by_area or self._stdev_scaled_by_area:
            scalefactor = self.calculate_scale_factor(modelpath)
        if self._rmsd_pure:
            result[i, 0] = rmsd
            i += 1
        if self._rmsd_scaled_bb_diag:
            result[i, 0] = rmsd / bbscale
            i += 1
        # samples = None
        # if self._rmsd_scaled_by_sampling:
        #     mesh = trimesh.load_mesh(modelpath)
        #     samples, _ = trimesh.sample.sample_surface(mesh, self._samplepoints)
        #     samples = torch.from_numpy(samples)
        #     rmsdref = pyeval.eval_rmsd_unscaled(pcbatch.view(-1, 3), samples)
        #     result[i, 0] = rmsd / rmsdref
        #     i += 1
        if self._rmsd_scaled_by_area:
            mesh = trimesh.load_mesh(modelpath)
            result[i, 0] = rmsd * scalefactor
            i += 1
        if self._md_pure:
            result[i, 0] = md
            i += 1
        if self._md_scaled_bb_diag:
            result[i, 0] = md / bbscale
            i += 1
        if self._md_scaled_by_area:
            result[i, 0] = md * scalefactor
            i += 1
        if self._stdev:
            result[i, 0] = stdev
            i += 1
        if self._stdev_scaled_bb_diag:
            result[i, 0] = stdev / bbscale
            i += 1
        if self._stdev_scaled_by_area:
            result[i, 0] = stdev * scalefactor
            i += 1
        if self._psnr:
            result[i, 0] = 20*math.log10(bbscale / rmsd)
            i += 1
        return result

    def get_names(self) -> List[str]:
        nlst = []
        if self._rmsd_pure:
            nlst.append("RMSD")
        if self._rmsd_scaled_bb_diag:
            nlst.append("RMSD scaled by BB")
        # if self._rmsd_scaled_by_sampling:
        #     nlst.append("RMSD scaled by Sampling")
        if self._rmsd_scaled_by_area:
            nlst.append("RMSD scaled by Area")
        if self._md_pure:
            nlst.append("MD")
        if self._md_scaled_bb_diag:
            nlst.append("MD scaled by BB")
        if self._md_scaled_by_area:
            nlst.append("MD scaled by Area")
        if self._stdev:
            nlst.append("MD Stdev")
        if self._stdev_scaled_bb_diag:
            nlst.append("MD Stdev scaled by BB")
        if self._stdev_scaled_by_area:
            nlst.append("MD Stdev scaled by Area")
        if self._psnr:
            nlst.append("PSNR")
        return nlst

    def calculate_scale_factor(self, modelpath: str):
       mesh = trimesh.load_mesh(modelpath)
       return 128 / math.sqrt(mesh.area)
