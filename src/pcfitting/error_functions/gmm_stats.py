from pcfitting import EvalFunction
from pcfitting.generators.em_tools import EMTools
from typing import List
import torch
import gmc.mixture as gm
import gmc.mat_tools as mat_tools
import math

class GMMStats(EvalFunction):
    # This class calculates metrics based on the statistics about the GMM by itself.

    def __init__(self,
                 avg_trace: bool = True,
                 stdev_traces: bool = True,
                 cv_traces: bool = True,
                 avg_evs: bool = True,
                 stdev_evs: bool = True,
                 min_ev: bool = True,
                 avg_amp: bool = True,
                 stdev_amp: bool = True,
                 avg_det: bool = True,
                 stdev_det: bool = True,
                 cv_det: bool = False,
                 avg_weight: bool = True,
                 stdev_weights: bool = True,
                 sum_of_weights: bool = True,
                 zero_gaussians: bool = True,
                 invalid_gaussians: bool = True,
                 em_default_abs_eps: bool = False,
                 avg_sqrt_det: bool = True,
                 std_sqrt_det: bool = True,
                 cv_ellvol: bool = True):
        # Parameters (if true, then...):
        # >> Note: All scaleable metrics are normalized by applying the pcbatch's attribute nnscalefactor.
        # Should be set before calling! <<
        # avg_trace: bool
        #   Calculates the average trace of the covariance matrices
        # stdev_traces: bool
        #   Calculates the standard deviation of traces of the covariance matrices
        # cv_traces: bool
        #   Calculates the coefficient of variation of traces of the covariance matrices
        # avg_evs: bool
        #   Calculates average eigenvalues (largest, medium and smallest respectively)
        # stdev_evs: bool
        #   Calculates standard deviations of eigenvalues (largest, medium, and smallest respectively)
        # avg_det: bool
        #   Calculates average determinant of the covariance matrices
        # stdev_det: bool
        #   Calculates standard deviation of the determinants covariance matrices
        # cv_det: bool
        #   Calculates coefficient of variation of the determinants
        # avg_weight: bool
        #   Calculates average prior weight
        # stdev_weights: bool
        #   Calculates standard deviation of prior weights
        # sum_of_weights: bool
        #   Calculates sum of prior weights (should be 1)
        # zero_gaussians: bool
        #   Calculates the number of Gaussians with prior weight = 0
        # invalid_gaussians: bool
        #   Calculates the number of Gaussians with non positive definite symmetric covariance matrix
        # em_default_abs_eps: bool
        #   Calculates absolute regularization value if relative value is 1e-7
        # avg_sqrt_det: bool
        #   Calculates average square root of determinant
        # std_sqrt_det: bool
        #   Calculates standard deviation of roots of determinants
        # cv_ellvol: bool
        #   Calculates coefficient of variation of isoellipsoid volumes by std_sqrt_det / avg_sqrt_det
        self._avg_trace = avg_trace
        self._stdev_traces = stdev_traces
        self._cv_traces = cv_traces
        self._avg_evs = avg_evs
        self._stdev_evs = stdev_evs
        self._min_ev = min_ev
        self._avg_amp = avg_amp
        self._stdev_amp = stdev_amp
        self._avg_det = avg_det
        self._stdev_det = stdev_det
        self._cv_det = cv_det
        self._avg_weight = avg_weight
        self._stdev_weights = stdev_weights
        self._sum_of_weights = sum_of_weights
        self._zero_gaussians = zero_gaussians
        self._invalid_gaussians = invalid_gaussians
        self._em_default_abs_eps = em_default_abs_eps
        self._avg_sqrt_det = avg_sqrt_det
        self._std_sqrt_det = std_sqrt_det
        self._cv_ellvol = cv_ellvol
        self._n_activated = avg_trace + stdev_traces + cv_traces + avg_evs*3 + stdev_evs*3 + min_ev + avg_amp + \
                            stdev_amp + avg_det + stdev_det + cv_det + \
                            avg_weight + stdev_weights + sum_of_weights + zero_gaussians + invalid_gaussians + \
                            em_default_abs_eps + avg_sqrt_det + std_sqrt_det + cv_ellvol

    def calculate_score(self, pcbatch: torch.Tensor, gmpositions: torch.Tensor, gmcovariances: torch.Tensor,
                        gminvcovariances: torch.Tensor, gmamplitudes: torch.Tensor,
                        noisecontribution: torch.Tensor = None, modelpath: str = None) -> torch.Tensor:
        result = torch.zeros(self._n_activated, pcbatch.shape[0], device=pcbatch.device, dtype=pcbatch.dtype)
        i = 0
        gmcov_filtered = gmcovariances[(~gmamplitudes.eq(0))].unsqueeze(0).unsqueeze(0)
        if self._avg_trace or self._stdev_traces or self._cv_traces:
            traces = mat_tools.trace(gmcov_filtered) # (bs, 1, ng)
            (std, mean) = torch.std_mean(traces, dim=2, unbiased=False)
            if self._avg_trace:
                result[i, :] = mean.view(-1) * (pcbatch.nnscalefactor ** 2)
                i += 1
            if self._stdev_traces:
                result[i, :] = std.view(-1) * (pcbatch.nnscalefactor ** 2)
                i += 1
            if self._cv_traces:
                result[i, :] = std.view(-1) / mean.view(-1)
                i += 1
        if self._avg_evs or self._stdev_evs or self._min_ev:
            evs, _ = torch.symeig(gmcov_filtered)
            (std, mean) = torch.std_mean(evs[:, 0, :, :], dim=1, unbiased=False) # (bs, 3)
            if self._avg_evs:
                result[i, :] = mean[:, 2] * pcbatch.nnscalefactor
                result[i + 1, :] = mean[:, 1] * pcbatch.nnscalefactor
                result[i + 2, :] = mean[:, 0] * pcbatch.nnscalefactor
                i += 3
            if self._stdev_evs:
                result[i, :] = std[:, 2] * pcbatch.nnscalefactor
                result[i + 1, :] = std[:, 1] * pcbatch.nnscalefactor
                result[i + 2, :] = std[:, 0] * pcbatch.nnscalefactor
                i += 3
            if self._min_ev:
                result[i, :] = torch.min(evs[:, 0, :, 0], dim=1)[0] * pcbatch.nnscalefactor
                i += 1
        if self._avg_amp or self._stdev_amp:
            (std, mean) = torch.std_mean(gmamplitudes[:, 0, :], dim=1, unbiased=False) # (bs)
            if self._avg_amp:
                result[i, :] = mean
                i += 1
            if self._stdev_amp:
                result[i, :] = std
                i += 1
        if self._avg_det or self._stdev_det or self._cv_det:
            (std, mean) = torch.std_mean(gmcov_filtered.det()[:, 0, :], dim=1, unbiased=False) # (bs)
            if self._avg_det:
                result[i, :] = mean * (pcbatch.nnscalefactor ** 6)
                i += 1
            if self._stdev_det:
                result[i, :] = std * (pcbatch.nnscalefactor ** 6)
                i += 1
            if self._cv_det:
                result[i, :] = std / mean
                i += 1
        if self._avg_weight or self._stdev_weights or self._sum_of_weights:
            weights = gmamplitudes * (gmcovariances.det().sqrt() * 15.74960995)
            (std, mean) = torch.std_mean(weights[:, 0, :], dim=1, unbiased=False)
            if self._avg_weight:
                result[i, :] = mean
                i += 1
            if self._stdev_weights:
                result[i, :] = std
                i += 1
            if self._sum_of_weights:
                result[i, :] = weights.sum(dim=2).view(-1)
                i += 1
        if self._zero_gaussians:
            result[i, :] = gmamplitudes.eq(0).sum(dim=2).view(-1)
            i += 1
        if self._invalid_gaussians:
            irelcovs = ~EMTools.find_valid_matrices(gmcovariances, gminvcovariances, False)
            result[i, :] = irelcovs.sum(dim=2).view(-1)
            i += 1
        if self._em_default_abs_eps:
            eps = 1e-7 * ((pcbatch[0].max(dim=0)[0] - pcbatch[0].min(dim=0)[0]).max(dim=0)[0].item() ** 2)
            if eps < 1e-9:
                eps = 1e-9
            result[i, :] = eps
            i += 1
        if self._avg_sqrt_det or self._std_sqrt_det or self._cv_ellvol:
            (std, mean) = torch.std_mean(gmcov_filtered.det().sqrt()[:, 0, :], dim=1, unbiased=False) # (bs)  # (bs, 1, ng)
            if self._avg_sqrt_det:
                result[i, :] = mean.view(-1) * (pcbatch.nnscalefactor ** 3)
                i += 1
            if self._std_sqrt_det:
                result[i, :] = std.view(-1) * (pcbatch.nnscalefactor ** 3)
                i += 1
            if self._cv_ellvol:
                result[i, :] = std.view(-1) / mean.view(-1)
                i += 1

        return result

    def get_names(self) -> List[str]:
        nlst = []
        if self._avg_trace:
            nlst.append("Average Trace")
        if self._stdev_traces:
            nlst.append("Stdev of Traces")
        if self._cv_traces:
            nlst.append("CV of Traces")
        if self._avg_evs:
            nlst.append("Average largest Eigenvalue")
            nlst.append("Average medium Eigenvalue")
            nlst.append("Average smallest Eigenvalue")
        if self._stdev_evs:
            nlst.append("Stdev of largest Eigenvalue")
            nlst.append("Stdev of medium Eigenvalue")
            nlst.append("Stdev of smallest Eigenvalue")
        if self._min_ev:
            nlst.append("Min Eigenvalue")
        if self._avg_amp:
            nlst.append("Average Amplitude")
        if self._stdev_amp:
            nlst.append("Stdev of Amplitudes")
        if self._avg_det:
            nlst.append("Average Determinant")
        if self._stdev_det:
            nlst.append("Stdev of Determinants")
        if self._cv_det:
            nlst.append("CV of Determinants")
        if self._avg_weight:
            nlst.append("Average Weight")
        if self._stdev_weights:
            nlst.append("Stdev of Weights")
        if self._sum_of_weights:
            nlst.append("Sum of Weights")
        if self._zero_gaussians:
            nlst.append("Number of 0-Gaussians")
        if self._invalid_gaussians:
            nlst.append("Number of invalid Gaussians")
        if self._em_default_abs_eps:
            nlst.append("EM default abs eps")
        if self._avg_sqrt_det:
            nlst.append("Average Sqrt Det")
        if self._std_sqrt_det:
            nlst.append("Stdev of Sqrt Det")
        if self._cv_ellvol:
            nlst.append("CV of Ellipsoid Volume")
        return nlst

    def needs_pc(self) -> bool:
        return False