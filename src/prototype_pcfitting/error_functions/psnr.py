import math

from prototype_pcfitting import ErrorFunction, GMSampler
import torch
import gmc.mixture as gm


class PSNR(ErrorFunction):
    # Calculates a score by calculating the PSNR of the point cloud given the mixture

    def __init__(self, nolog: bool = False):
        self._nolog = nolog

    def calculate_score(self, pcbatch: torch.Tensor, gmpositions: torch.Tensor, gmcovariances: torch.Tensor,
                        gminvcovariances: torch.Tensor, gmamplitudes: torch.Tensor,
                        noisecontribution: torch.Tensor = None) -> torch.Tensor:
        batch_size = pcbatch.shape[0]
        point_size = pcbatch.shape[1]
        bbmin = pcbatch.min(dim=1)[0]
        bbmax = pcbatch.max(dim=1)[0]
        bbsizes = (bbmax - bbmin).norm(dim=1)
        gmm = gm.convert_amplitudes_to_priors(gm.pack_mixture(gmamplitudes, gmpositions, gmcovariances))
        sampled = GMSampler.sample(gmm, point_size)
        minsqdiffs = torch.zeros(batch_size, point_size, dtype=pcbatch.dtype, device=pcbatch.device)
        point_batch = 1000
        for i_start in range(0, point_size, point_batch):
            i_len = min(point_size - i_start, point_batch)
            i_end = i_start + i_len
            pcbatch_rep = pcbatch[:,i_start:i_end].unsqueeze(2).unsqueeze(-1).expand(batch_size, i_len, point_size, 3, 1)
            sampled_rep = sampled.unsqueeze(1).unsqueeze(-1).expand(batch_size, i_len, point_size, 3, 1)
            diffs = (pcbatch_rep - sampled_rep)
            sqdiffs = (diffs.transpose(-1, -2) @ diffs).squeeze(-1).squeeze(-1) # shape: (bs, il, ps)
            minsqdiffs[:, i_start:i_end] = sqdiffs.min(dim=2)[0]
        avgminsqdiff = minsqdiffs.mean(dim=1) # shape: (bs)
        relrms = (bbsizes) / torch.sqrt(avgminsqdiff)
        if not self._nolog:
            relrms = 20*torch.log10(relrms) # https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        return relrms
