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
                 rmsd_pure: bool = False,
                 rmsd_scaled_bb_diag: bool = False,
                 rmsd_scaled_by_area: bool = True,
                 md_pure: bool = False,
                 md_scaled_bb_diag: bool = False,
                 md_scaled_by_area: bool = True,
                 stdev: bool = False,
                 stdev_scaled_bb_diag: bool = False,
                 stdev_scaled_by_area: bool = True,
                 stdev_mean_ratio: bool = False,
                 psnr: bool = False,
                 maxdist: bool = False,
                 maxdist_norm: bool = True,
                 inverse: bool = True,
                 chamfer: bool = False,
                 chamfer_norm: bool = True,
                 hausdorff: bool = False,
                 hausdorff_norm: bool = True,
                 sample_points: int = 100000):
        self._rmsd_pure = rmsd_pure
        self._rmsd_scaled_bb_diag = rmsd_scaled_bb_diag
        self._rmsd_scaled_by_area = rmsd_scaled_by_area
        self._md_pure = md_pure
        self._md_scaled_bb_diag = md_scaled_bb_diag
        self._md_scaled_by_area = md_scaled_by_area
        self._stdev = stdev
        self._stdev_scaled_bb_diag = stdev_scaled_bb_diag
        self._stdev_scaled_by_area = stdev_scaled_by_area
        self._stdev_mean_ratio = stdev_mean_ratio
        self._psnr = psnr
        self._maxdist = maxdist
        self._maxdist_norm = maxdist_norm
        self._inverse = inverse
        self._chamfer = chamfer
        self._chamfer_norm = chamfer_norm
        self._hausdorff = hausdorff
        self._hausdorff_norm = hausdorff_norm
        self._samplepoints = sample_points
        self._nmeth = (self._rmsd_pure + self._rmsd_scaled_bb_diag + self._rmsd_scaled_by_area + self._md_pure
                + self._md_scaled_bb_diag + self._md_scaled_by_area + self._stdev
                + self._stdev_scaled_bb_diag + self._stdev_scaled_by_area + self._stdev_mean_ratio + self._psnr
                       + self._maxdist + self._maxdist_norm) *\
                (2 if self._inverse else 1 ) + 2*(self._chamfer + self._chamfer_norm + self._hausdorff + self._hausdorff_norm)
        pass

    def calculate_score(self, pcbatch: torch.Tensor, gmpositions: torch.Tensor, gmcovariances: torch.Tensor,
                        gminvcovariances: torch.Tensor, gmamplitudes: torch.Tensor,
                        noisecontribution: torch.Tensor = None, modelpath: str = None) -> torch.Tensor:
        batch_size = pcbatch.shape[0]
        assert batch_size == 1

        result = torch.zeros(self._nmeth, batch_size, device=pcbatch.device, dtype=pcbatch.dtype)

        gmm = gm.convert_amplitudes_to_priors(gm.pack_mixture(gmamplitudes, gmpositions, gmcovariances))
        sampled = GMSampler.sample(gmm, self._samplepoints)

        pmin = torch.min(pcbatch[0], dim=0)[0]
        pmax = torch.max(pcbatch[0], dim=0)[0]
        bbscale = torch.norm(pmax - pmin).item()

        i = 0
        rmsd, md, stdev, maxd = pyeval.eval_rmsd_unscaled(pcbatch.view(-1, 3), sampled.view(-1, 3))
        rmsdI, mdI, stdevI, maxdI = (None, None, None, None)
        if self._inverse or self._chamfer or self._chamfer_norm:
            rmsdI, mdI, stdevI, maxdI = pyeval.eval_rmsd_unscaled(sampled.view(-1, 3), pcbatch.view(-1, 3))
        scalefactor = 1
        if self._stdev_scaled_by_area or self._md_scaled_by_area or self._stdev_scaled_by_area:
            scalefactor = self.calculate_scale_factor(modelpath)
        if self._rmsd_pure:
            result[i, 0] = rmsd
            i += 1
            if self._inverse:
                result[i, 0] = rmsdI
                i += 1
        if self._rmsd_scaled_bb_diag:
            result[i, 0] = rmsd / bbscale
            i += 1
            if self._inverse:
                result[i, 0] = rmsdI / bbscale
                i += 1
        if self._rmsd_scaled_by_area:
            mesh = trimesh.load_mesh(modelpath)
            result[i, 0] = rmsd * scalefactor
            i += 1
            if self._inverse:
                result[i, 0] = rmsdI * scalefactor
                i += 1
        if self._md_pure:
            result[i, 0] = md
            i += 1
            if self._inverse:
                result[i, 0] = mdI
                i += 1
        if self._md_scaled_bb_diag:
            result[i, 0] = md / bbscale
            i += 1
            if self._inverse:
                result[i, 0] = mdI / bbscale
                i += 1
        if self._md_scaled_by_area:
            result[i, 0] = md * scalefactor
            i += 1
            if self._inverse:
                result[i, 0] = mdI * scalefactor
                i += 1
        if self._stdev:
            result[i, 0] = stdev
            i += 1
            if self._inverse:
                result[i, 0] = stdevI
                i += 1
        if self._stdev_scaled_bb_diag:
            result[i, 0] = stdev / bbscale
            i += 1
            if self._inverse:
                result[i, 0] = stdevI / bbscale
                i += 1
        if self._stdev_scaled_by_area:
            result[i, 0] = stdev * scalefactor
            i += 1
            if self._inverse:
                result[i, 0] = stdevI * scalefactor
                i += 1
        if self._stdev_mean_ratio:
            result[i, 0] = stdev / md
            i += 1
            if self._inverse:
                result[i, 0] = stdevI / mdI
                i += 1
        if self._psnr:
            result[i, 0] = 20*math.log10(bbscale / rmsd)
            i += 1
            if self._inverse:
                result[i, 0] = 20*math.log10(bbscale / rmsdI)
                i += 1
        if self._maxdist:
            result[i, 0] = maxd
            i += 1
            if self._inverse:
                result[i, 0] = maxdI
                i += 1
        if self._maxdist_norm:
            result[i, 0] = maxd * scalefactor
            i += 1
            if self._inverse:
                result[i, 0] = maxdI * scalefactor
                i += 1
        if self._chamfer:
            result[i, 0] = (rmsd**2) + (rmsdI**2)
            result[i+1, 0] = math.sqrt(result[i, 0])
            i += 2
        if self._chamfer_norm:
            chamf = (rmsd**2) + (rmsdI**2)
            result[i, 0] = chamf * (scalefactor**2)
            result[i+1, 0] = math.sqrt(chamf) * scalefactor
            i += 2
        if self._hausdorff:
            result[i, 0] = max(maxd, maxdI)
            i += 1
        if self._hausdorff_norm:
            result[i, 0] = max(maxd, maxdI) * scalefactor
            i += 1
        return result

    def get_names(self) -> List[str]:
        nlst = []
        if self._rmsd_pure:
            nlst.append("RMSD")
            if self._inverse:
                nlst.append("Inverse RMSD")
        if self._rmsd_scaled_bb_diag:
            nlst.append("RMSD scaled by BB")
            if self._inverse:
                nlst.append("Inverse RMSD scaled by BB")
        # if self._rmsd_scaled_by_sampling:
        #     nlst.append("RMSD scaled by Sampling")
        if self._rmsd_scaled_by_area:
            nlst.append("RMSD norm.")
            if self._inverse:
                nlst.append("Inverse RMSD norm.")
        if self._md_pure:
            nlst.append("MD")
            if self._inverse:
                nlst.append("Inverse MD")
        if self._md_scaled_bb_diag:
            nlst.append("MD scaled by BB")
            if self._inverse:
                nlst.append("Inverse MD scaled by BB")
        if self._md_scaled_by_area:
            nlst.append("MD norm.")
            if self._inverse:
                nlst.append("Inverse MD norm.")
        if self._stdev:
            nlst.append("MD Stdev")
            if self._inverse:
                nlst.append("Inverse MD Stdev")
        if self._stdev_scaled_bb_diag:
            nlst.append("MD Stdev scaled by BB")
            if self._inverse:
                nlst.append("Inverse MD Stdev scaled by BB")
        if self._stdev_scaled_by_area:
            nlst.append("MD Stdev norm.")
            if self._inverse:
                nlst.append("Inverse MD Stdev norm.")
        if self._stdev_mean_ratio:
            nlst.append("MD Stdev-Mean-Ratio")
            if self._inverse:
                nlst.append("Inverse MD Stdev-Mean-Ratio")
        if self._psnr:
            nlst.append("PSNR")
            if self._inverse:
                nlst.append("Inverse PSNR")
        if self._maxdist:
            nlst.append("Maxdist")
            if self._inverse:
                nlst.append("Inverse Maxdist")
        if self._maxdist_norm:
            nlst.append("Maxdist norm.")
            if self._inverse:
                nlst.append("Inverse Maxdist norm.")
        if self._chamfer:
            nlst.append("Chamfer Distance")
            nlst.append("Root of Chamfer Distance")
        if self._chamfer_norm:
            nlst.append("Chamfer Distance Norm.")
            nlst.append("Root of Chamfer Distance Norm.")
        if self._hausdorff:
            nlst.append("Hausdorff Distance")
        if self._hausdorff_norm:
            nlst.append("Hausdorff Distance Norm")
        return nlst

    def calculate_scale_factor(self, modelpath: str):
       mesh = trimesh.load_mesh(modelpath)
       return 128 / math.sqrt(mesh.area)
