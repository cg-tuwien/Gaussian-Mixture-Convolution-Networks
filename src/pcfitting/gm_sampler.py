import torch
import gmc.mixture
import numpy
from pcfitting.cpp.gmeval import pyeval

class GMSampler:

    @staticmethod
    def sampleGM(gm: torch.Tensor, count: int):
        gmm = gmc.mixture.convert_amplitudes_to_priors(gm)
        return GMSampler.sampleGMM(gmm, count)

    @staticmethod
    def sampleGMM(gmm: torch.Tensor, count: int):
        # Samples count points from the given mixture model
        # result size: [bs,n,3]
        batch_size = gmm.shape[0]
        samples = torch.zeros(batch_size, count, 3, dtype=gmm.dtype, device=gmm.device)
        uniformprobs = torch.from_numpy(numpy.random.uniform(size=(batch_size, count))).to(gmm.device)
        means = gmc.mixture.positions(gmm)
        covs = gmc.mixture.covariances(gmm)
        weights = gmc.mixture.weights(gmm).clone()
        gauss_count = weights.shape[2]
        for i in range(1, gauss_count):
            weights[:,0,i] += weights[:,0,i-1]
        gaussidzs = torch.zeros(batch_size, count, dtype=torch.int)
        uniformprobs *= weights[:,0,-1]
        for b in range(batch_size):
            for i in range(count):
                gaussidzs[b, i] = torch.nonzero(uniformprobs[b, i] <= weights[b,0,:])[0]
            for g in range(gauss_count):
                selected_samples = gaussidzs[b, :].eq(g)
                if selected_samples.sum().item() > 0:
                    samples[b, selected_samples, :] = torch.tensor(numpy.random.multivariate_normal(means[b,0,g].cpu(), covs[b,0,g].cpu(), selected_samples.sum().item()), dtype=gmm.dtype, device=gmm.device)
        return samples

    @staticmethod
    def sampleGMM_ext(gmm: torch.Tensor, count: int) -> torch.Tensor:
        return pyeval.sample_gmm(gmm[0, 0].cpu(), count).to(gmm.device)