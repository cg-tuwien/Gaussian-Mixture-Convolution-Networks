import torch
import gmc.mixture
import numpy


class GMSampler:

    @staticmethod
    def sample(gmm: torch.Tensor, count: int):
        # Samples count points from the given mixture model
        # result size: [1,n,3]
        samples = torch.zeros(1, count, 3)
        uniformprobs = numpy.random.uniform(size=count)
        means = gmc.mixture.positions(gmm)
        covs = gmc.mixture.covariances(gmm)
        weights = gmc.mixture.weights(gmm).clone()
        for i in range(1, weights.shape[2]):
            weights[0,0,i] += weights[0,0,i-1]
        gaussidzs = torch.zeros(count, dtype=torch.int)
        for i in range(count):
            gaussidzs[i] = torch.nonzero(uniformprobs[i] < weights[0,0,:])[0]
        for i in range(count):
            idx = gaussidzs[i]
            samples[0,i,:] = torch.tensor(numpy.random.multivariate_normal(means[0,0,idx], covs[0,0,idx]))
        return samples
