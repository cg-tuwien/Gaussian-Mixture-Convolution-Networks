import torch
import gmc.mixture
import numpy


class GMSampler:

    @staticmethod
    def sample(gmm: torch.Tensor, count: int):
        # Samples count points from the given mixture model
        # result size: [bs,n,3]
        batch_size = gmm.shape[0]
        samples = torch.zeros(batch_size, count, 3, dtype=gmm.dtype, device=gmm.device)
        uniformprobs = numpy.random.uniform(size=(batch_size, count))
        means = gmc.mixture.positions(gmm)
        covs = gmc.mixture.covariances(gmm)
        weights = gmc.mixture.weights(gmm).clone()
        for i in range(1, weights.shape[2]):
            weights[:,0,i] += weights[:,0,i-1]
        gaussidzs = torch.zeros(batch_size, count, dtype=torch.int)
        for b in range(batch_size):
            for i in range(count):
                gaussidzs[b, i] = torch.nonzero(uniformprobs[b, i] < weights[b,0,:])[0]
            for i in range(count):
                idx = gaussidzs[b, i]
                samples[b,i,:] = torch.tensor(numpy.random.multivariate_normal(means[b,0,idx].cpu(), covs[b,0,idx].cpu()), dtype=gmm.dtype, device=gmm.device)
        return samples
