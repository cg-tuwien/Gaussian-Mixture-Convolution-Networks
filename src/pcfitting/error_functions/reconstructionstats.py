import math
from typing import List
import time

import trimesh
import trimesh.sample
import trimesh.proximity

from pcfitting import EvalFunction, GMSampler
import torch
import numpy as np
import gmc.mixture as gm
import gmc.mat_tools as mat_tools
from pcfitting.cpp.gmeval import pyeval


class ReconstructionStats(EvalFunction):

    def __init__(self,
                 rmsd_pure: bool = False,
                 rmsd_scaled_bb_diag: bool = False,
                 rmsd_scaled_by_area: bool = False,
                 rmsd_scaled_by_nn: bool = True,
                 md_pure: bool = False,
                 md_scaled_bb_diag: bool = False,
                 md_scaled_by_area: bool = False, #
                 md_scaled_by_nn: bool = True,
                 stdev: bool = False,
                 stdev_scaled_bb_diag: bool = False,
                 stdev_scaled_by_area: bool = False, #
                 stdev_scaled_by_nn: bool = True,
                 cv: bool = True, #
                 kurtosis: bool = False, #
                 psnr: bool = False,
                 maxdist: bool = False,
                 maxdist_norm_area: bool = False,
                 maxdist_norm_nn: bool = False,
                 inverse: bool = True,
                 inverse_exact: bool = False,
                 chamfer: bool = False,
                 chamfer_norm_area: bool = False,
                 chamfer_norm_nn: bool = False,
                 hausdorff: bool = False,
                 hausdorff_norm_area: bool = False,
                 hausdorff_norm_nn: bool = False,
                 cov_measure: bool = False, #
                 cov_measure_scaled_by_area: bool = False,
                 sample_points: int = 100000):
        self._rmsd_pure = rmsd_pure
        self._rmsd_scaled_bb_diag = rmsd_scaled_bb_diag
        self._rmsd_scaled_by_area = rmsd_scaled_by_area
        self._rmsd_scaled_by_nn = rmsd_scaled_by_nn
        self._md_pure = md_pure
        self._md_scaled_bb_diag = md_scaled_bb_diag
        self._md_scaled_by_area = md_scaled_by_area
        self._md_scaled_by_nn = md_scaled_by_nn
        self._stdev = stdev
        self._stdev_scaled_bb_diag = stdev_scaled_bb_diag
        self._stdev_scaled_by_area = stdev_scaled_by_area
        self._stdev_scaled_by_nn = stdev_scaled_by_nn
        self._cv = cv
        self._kurtosis = kurtosis
        self._psnr = psnr
        self._maxdist = maxdist
        self._maxdist_norm_area = maxdist_norm_area
        self._maxdist_norm_nn = maxdist_norm_nn
        self._inverse = inverse
        self._inverse_exact = inverse_exact
        self._chamfer = chamfer
        self._chamfer_norm_area = chamfer_norm_area
        self._chamfer_norm_nn = chamfer_norm_nn
        self._hausdorff = hausdorff
        self._hausdorff_norm_area = hausdorff_norm_area
        self._hausdorff_norm_nn = hausdorff_norm_nn
        self._samplepoints = sample_points
        self._cov_measure = cov_measure
        self._cov_measure_std_scaled_by_area = cov_measure_scaled_by_area
        self._nmeth = (self._rmsd_pure + self._rmsd_scaled_bb_diag + self._rmsd_scaled_by_area +
                       self._rmsd_scaled_by_nn + self._md_pure + self._md_scaled_bb_diag + self._md_scaled_by_area +
                       self._md_scaled_by_nn + self._stdev + self._stdev_scaled_bb_diag + self._stdev_scaled_by_area +
                       self._stdev_scaled_by_nn + self._cv + self._psnr + self._maxdist + self._maxdist_norm_area
                       + self._maxdist_norm_nn + self._kurtosis) *\
                        (2 if self._inverse else 1 ) + 2*(self._chamfer + self._chamfer_norm_area + self._chamfer_norm_nn) + \
                        self._hausdorff + self._hausdorff_norm_area + self._hausdorff_norm_nn + self._cov_measure * 2 + self._cov_measure_std_scaled_by_area
        pass

    def calculate_score_on_reconstructed(self, pcbatch: torch.Tensor, sampled: torch.Tensor,
                                         modelpath: str = None) -> torch.Tensor:
        batch_size = pcbatch.shape[0]
        assert batch_size == 1

        # start = time.time()

        result = torch.zeros(self._nmeth, batch_size, device=pcbatch.device, dtype=pcbatch.dtype)

        pmin = torch.min(pcbatch[0], dim=0)[0]
        pmax = torch.max(pcbatch[0], dim=0)[0]
        bbscale = torch.norm(pmax - pmin).item()

        i = 0
        if (self._inverse or self._chamfer or self._chamfer_norm_area or self._chamfer_norm_nn) and not self._inverse_exact:
            rmsd, md, stdev, maxd, rmsdI, mdI, stdevI, maxdI = pyeval.eval_rmsd_both_sides(pcbatch.view(-1, 3), sampled.view(-1, 3))
        else:
            rmsd, md, stdev, maxd = pyeval.eval_rmsd_unscaled(pcbatch.view(-1, 3), sampled.view(-1, 3))
            rmsdI, mdI, stdevI, maxdI = (None, None, None, None)
            if self._inverse or self._chamfer or self._chamfer_norm_area or self._chamfer_norm_nn:
                if self._inverse_exact:
                    rmsdI, mdI, stdevI, maxdI = self.calc_inverse_exact(sampled.view(-1, 3), modelpath)
                else:
                    rmsdI, mdI, stdevI, maxdI = pyeval.eval_rmsd_unscaled(sampled.view(-1, 3), pcbatch.view(-1, 3))
        scalefactor = 1
        if self._stdev_scaled_by_area or self._md_scaled_by_area or self._stdev_scaled_by_area or \
                self._chamfer_norm_area or self._hausdorff_norm_area or self._cov_measure_std_scaled_by_area:
            scalefactor = self.calculate_scale_factor(modelpath)
        scalefactorNN = 1
        if self._stdev_scaled_by_nn or self._md_scaled_by_nn or self._stdev_scaled_by_nn or self._chamfer_norm_nn or \
                self._hausdorff_norm_nn:
            scalefactorNN = self.calculate_scale_factor_nn(pcbatch)
        if self._rmsd_pure:
            result[i, 0] = rmsd
            result.rmsd_pure = result[i, 0].item()
            i += 1
            if self._inverse:
                result[i, 0] = rmsdI
                result.rmsd_pure_I = result[i, 0].item()
                i += 1
        if self._rmsd_scaled_bb_diag:
            result[i, 0] = rmsd / bbscale
            result.rmsd_scaled_bb_diag = result[i, 0].item()
            i += 1
            if self._inverse:
                result[i, 0] = rmsdI / bbscale
                result.rmsd_scaled_bb_diag_I = result[i, 0].item()
                i += 1
        if self._rmsd_scaled_by_area:
            result[i, 0] = rmsd * scalefactor
            result.rmsd_scaled_by_area = result[i, 0].item()
            i += 1
            if self._inverse:
                result[i, 0] = rmsdI * scalefactor
                result.rmsd_scaled_by_area_I = result[i, 0].item()
                i += 1
        if self._rmsd_scaled_by_nn:
            result[i, 0] = rmsd * scalefactorNN
            result.rmsd_scaled_by_nn = result[i, 0].item()
            i += 1
            if self._inverse:
                result[i, 0] = rmsdI * scalefactorNN
                result.rmsd_scaled_by_nn_I = result[i, 0].item()
                i += 1
        if self._md_pure:
            result[i, 0] = md
            result.md_pure = result[i, 0].item()
            i += 1
            if self._inverse:
                result[i, 0] = mdI
                result.md_pure_I = result[i, 0].item()
                i += 1
        if self._md_scaled_bb_diag:
            result[i, 0] = md / bbscale
            result.md_scaled_bb_diag = result[i, 0].item()
            i += 1
            if self._inverse:
                result[i, 0] = mdI / bbscale
                result.md_scaled_bb_diag_I = result[i, 0].item()
                i += 1
        if self._md_scaled_by_area:
            result[i, 0] = md * scalefactor
            result.md_scaled_by_area = result[i, 0].item()
            i += 1
            if self._inverse:
                result[i, 0] = mdI * scalefactor
                result.md_scaled_by_area_I = result[i, 0].item()
                i += 1
        if self._md_scaled_by_nn:
            result[i, 0] = md * scalefactorNN
            result.md_scaled_by_nn = result[i, 0].item()
            i += 1
            if self._inverse:
                result[i, 0] = mdI * scalefactorNN
                result.md_scaled_by_nn_I = result[i, 0].item()
                i += 1
        if self._stdev:
            result[i, 0] = stdev
            result.stdev = result[i, 0].item()
            i += 1
            if self._inverse:
                result[i, 0] = stdevI
                result.stdev_I = result[i, 0].item()
                i += 1
        if self._stdev_scaled_bb_diag:
            result[i, 0] = stdev / bbscale
            result.stdev_scaled_bb_diag = result[i, 0].item()
            i += 1
            if self._inverse:
                result[i, 0] = stdevI / bbscale
                result.stdev_scaled_bb_diag_I = result[i, 0].item()
                i += 1
        if self._stdev_scaled_by_area:
            result[i, 0] = stdev * scalefactor
            result.stdev_scaled_by_area = result[i, 0].item()
            i += 1
            if self._inverse:
                result[i, 0] = stdevI * scalefactor
                result.stdev_scaled_by_area_I = result[i, 0].item()
                i += 1
        if self._stdev_scaled_by_nn:
            result[i, 0] = stdev * scalefactorNN
            result.stdev_scaled_by_nn = result[i, 0].item()
            i += 1
            if self._inverse:
                result[i, 0] = stdevI * scalefactorNN
                result.stdev_scaled_by_nn_I = result[i, 0].item()
                i += 1
        if self._cv:
            result[i, 0] = stdev / md
            result.cv = result[i, 0].item()
            i += 1
            if self._inverse:
                result[i, 0] = stdevI / mdI
                result.cv_I = result[i, 0].item()
                i += 1
        if self._kurtosis:
            # currently disabled. change pyeval and add it as a return value of the calculations to use it again
            result[i, 0] = kurtosis
            result.kurtosis = result[i, 0].item()
            i += 1
            if self._inverse:
                result[i, 0] = kurtosisI
                result.kurtosisI = result[i, 0].item()
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
        if self._maxdist_norm_area:
            result[i, 0] = maxd * scalefactor
            i += 1
            if self._inverse:
                result[i, 0] = maxdI * scalefactor
                i += 1
        if self._maxdist_norm_nn:
            result[i, 0] = maxd * scalefactorNN
            i += 1
            if self._inverse:
                result[i, 0] = maxdI * scalefactorNN
                i += 1
        if self._chamfer:
            result[i, 0] = (rmsd**2) + (rmsdI**2)
            result[i+1, 0] = math.sqrt(result[i, 0])
            i += 2
        if self._chamfer_norm_area:
            chamf = (rmsd**2) + (rmsdI**2)
            result[i, 0] = chamf * (scalefactor**2)
            result[i+1, 0] = math.sqrt(chamf) * scalefactor
            i += 2
        if self._chamfer_norm_nn:
            chamf = (rmsd**2) + (rmsdI**2)
            result[i, 0] = chamf * (scalefactorNN**2)
            result[i+1, 0] = math.sqrt(chamf) * scalefactorNN
            i += 2
        if self._hausdorff:
            result[i, 0] = max(maxd, maxdI)
            i += 1
        if self._hausdorff_norm_area:
            result[i, 0] = max(maxd, maxdI) * scalefactor
            i += 1
        if self._hausdorff_norm_nn:
            result[i, 0] = max(maxd, maxdI) * scalefactorNN
            i += 1
        if self._cov_measure or self._cov_measure_std_scaled_by_area:
            (covm, covmstd) = pyeval.cov_measure(sampled.view(-1, 3))
            if self._cov_measure:
                result[i, 0] = covm
                result.cov_measure = result[i, 0].item()
                result[i+1, 0] = covmstd
                result.cov_measure_std = result[i+1, 0].item()
                i += 2
            if self._cov_measure_std_scaled_by_area:
                result[i, 0] = covmstd * scalefactor
                result.cov_measure_std_scaled_by_area = result[i, 0].item()
                i += 1

        # end = time.time()
        # print ("Time taken: ", (end-start))
        return result


    def calculate_score(self, pcbatch: torch.Tensor, gmpositions: torch.Tensor, gmcovariances: torch.Tensor,
                        gminvcovariances: torch.Tensor, gmamplitudes: torch.Tensor,
                        noisecontribution: torch.Tensor = None, modelpath: str = None) -> torch.Tensor:
        batch_size = pcbatch.shape[0]
        assert batch_size == 1

        gmm = gm.convert_amplitudes_to_priors(gm.pack_mixture(gmamplitudes, gmpositions, gmcovariances))
        sampled = GMSampler.sampleGMM(gmm, self._samplepoints)

        return self.calculate_score_on_reconstructed(pcbatch, sampled, modelpath)

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
            nlst.append("RMSD normAR")
            if self._inverse:
                nlst.append("Inverse RMSD normAR")
        if self._rmsd_scaled_by_nn:
            nlst.append ("RMSD normNN")
            if self._inverse:
                nlst.append("Inverse RMSD normAR")
        if self._md_pure:
            nlst.append("MD")
            if self._inverse:
                nlst.append("Inverse MD")
        if self._md_scaled_bb_diag:
            nlst.append("MD scaled by BB")
            if self._inverse:
                nlst.append("Inverse MD scaled by BB")
        if self._md_scaled_by_area:
            nlst.append("MD normAR")
            if self._inverse:
                nlst.append("Inverse MD normAR")
        if self._md_scaled_by_nn:
            nlst.append("MD normNN")
            if self._inverse:
                nlst.append("Inverse MD normNN")
        if self._stdev:
            nlst.append("MD Stdev")
            if self._inverse:
                nlst.append("Inverse MD Stdev")
        if self._stdev_scaled_bb_diag:
            nlst.append("MD Stdev scaled by BB")
            if self._inverse:
                nlst.append("Inverse MD Stdev scaled by BB")
        if self._stdev_scaled_by_area:
            nlst.append("MD Stdev normAR")
            if self._inverse:
                nlst.append("Inverse MD Stdev normAR")
        if self._stdev_scaled_by_nn:
            nlst.append("MD Stdev normNN")
            if self._inverse:
                nlst.append("Inverse MD Stdev normNN")
        if self._cv:
            nlst.append("MD CV")
            if self._inverse:
                nlst.append("Inverse MD CV")
        if self._psnr:
            nlst.append("PSNR")
            if self._inverse:
                nlst.append("Inverse PSNR")
        if self._kurtosis:
            nlst.append("Kurtosis")
            if self._inverse:
                nlst.append("Inverse Kurtosis")
        if self._maxdist:
            nlst.append("Maxdist")
            if self._inverse:
                nlst.append("Inverse Maxdist")
        if self._maxdist_norm_area:
            nlst.append("Maxdist normAR")
            if self._inverse:
                nlst.append("Inverse Maxdist normAR")
        if self._maxdist_norm_nn:
            nlst.append("Maxdist normNN")
            if self._inverse:
                nlst.append("Inverse Maxdist normNN")
        if self._chamfer:
            nlst.append("Chamfer Distance")
            nlst.append("Root of Chamfer Distance")
        if self._chamfer_norm_area:
            nlst.append("Chamfer Distance NormAR")
            nlst.append("Root of Chamfer Distance NormAR")
        if self._chamfer_norm_nn:
            nlst.append("Chamfer Distance NormNN")
            nlst.append("Root of Chamfer Distance NormNN")
        if self._hausdorff:
            nlst.append("Hausdorff Distance")
        if self._hausdorff_norm_area:
            nlst.append("Hausdorff Distance NormAR")
        if self._hausdorff_norm_nn:
            nlst.append("Hausdorff Distance NormNN")
        if self._cov_measure:
            nlst.append("COV measure")
            nlst.append("COV measure std")
        if self._cov_measure_std_scaled_by_area:
            nlst.append("COV measure std NormAR")
        return nlst

    def calculate_scale_factor(self, modelpath: str):
       mesh = trimesh.load_mesh(modelpath)
       return 128 / math.sqrt(mesh.area)

    def calculate_scale_factor_nn(self, pcbatch: torch.Tensor) -> float:
        if not hasattr(pcbatch, "nnscalefactor"):
            md = pyeval.calc_rmsd_to_itself(pcbatch.view(-1, 3))[1]
            refdist = 128 / (2*math.sqrt(pcbatch.shape[1]) - 1)
            pcbatch.nnscalefactor = refdist / md
        return pcbatch.nnscalefactor

    def calc_inverse_exact(self, pc: torch.Tensor, modelpath: str):
        mesh = trimesh.load_mesh(modelpath)
        query = trimesh.proximity.ProximityQuery(mesh)
        closest, distances, triangle_id = query.on_surface(pc.cpu().numpy())
        sqdistances = np.square(distances)
        rmsd = np.sqrt(np.mean(sqdistances))
        md = np.mean(distances)
        std = np.sqrt(np.sum(np.square((distances - md))) / (distances.shape[0] - 1))
        maxd = np.max(distances)
        return rmsd, md, std, maxd
