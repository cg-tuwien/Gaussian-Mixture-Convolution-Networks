from pcfitting import EvalFunction
from pcfitting.generators.em_tools import EMTools
from typing import List
import torch
import gmc.mixture as gm
import gmc.mat_tools as mat_tools

class GMMStats(EvalFunction):

    def __init__(self,
                 avg_trace: bool = True,
                 stdev_traces: bool = True,
                 avg_evs: bool = True,
                 stdev_evs: bool = True,
                 min_ev: bool = True,
                 avg_amp: bool = True,
                 stdev_amp: bool = True,
                 avg_weight: bool = True,
                 stdev_weights: bool = True,
                 sum_of_weights: bool = True,
                 zero_gaussians: bool = True,
                 invalid_gaussians: bool = True):
        self._avg_trace = avg_trace
        self._stdev_traces = stdev_traces
        self._avg_evs = avg_evs
        self._stdev_evs = stdev_evs
        self._min_ev = min_ev
        self._avg_amp = avg_amp
        self._stdev_amp = stdev_amp
        self._avg_weight = avg_weight
        self._stdev_weights = stdev_weights
        self._sum_of_weights = sum_of_weights
        self._zero_gaussians = zero_gaussians
        self._invalid_gaussians = invalid_gaussians
        self._n_activated = avg_trace + stdev_traces + avg_evs*3 + stdev_evs*3 + min_ev + avg_amp + stdev_amp + \
                            avg_weight + stdev_weights + sum_of_weights + zero_gaussians + invalid_gaussians

    def calculate_score(self, pcbatch: torch.Tensor, gmpositions: torch.Tensor, gmcovariances: torch.Tensor,
                        gminvcovariances: torch.Tensor, gmamplitudes: torch.Tensor,
                        noisecontribution: torch.Tensor = None) -> torch.Tensor:
        result = torch.zeros(self._n_activated, pcbatch.shape[0], device=pcbatch.device, dtype=pcbatch.dtype)
        i = 0
        if self._avg_trace or self._stdev_traces:
            traces = mat_tools.trace(gmcovariances) # (bs, 1, ng)
            (std, mean) = torch.std_mean(traces, dim=2, unbiased=False)
            if self._avg_trace:
                result[i, :] = mean.view(-1)
                i += 1
            if self._stdev_traces:
                result[i, :] = std.view(-1)
                i += 1
        if self._avg_evs or self._stdev_evs or self._min_ev:
            evs, _ = torch.symeig(gmcovariances)
            (std, mean) = torch.std_mean(evs[:, 0, :, :], dim=1, unbiased=False) # (bs, 3)
            if self._avg_evs:
                result[i, :] = mean[:, 2]
                result[i + 1, :] = mean[:, 1]
                result[i + 2, :] = mean[:, 0]
                i += 3
            if self._stdev_evs:
                result[i, :] = std[:, 2]
                result[i + 1, :] = std[:, 1]
                result[i + 2, :] = std[:, 0]
                i += 3
            if self._min_ev:
                result[i, :] = torch.min(evs[:, 0, :, 0], dim=1)[0]
                i += 1
        if self._avg_amp or self._stdev_amp:
            (std, mean) = torch.std_mean(gmamplitudes[:, 0, :], dim=1, unbiased=False) # (bs)
            if self._avg_amp:
                result[i, :] = mean
                i += 1
            if self._stdev_amp:
                result[i, :] = std
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
        return result

    def get_names(self) -> List[str]:
        nlst = []
        if self._avg_trace:
            nlst.append("Average Trace")
        if self._stdev_traces:
            nlst.append("Stdev of Traces")
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
        return nlst

    def needs_pc(self) -> bool:
        return False