import torch
from sklearn.cluster import KMeans

import gmc.mixture as gm
import numpy as np

from prototype_pcfitting import data_loading
from gmc.cpp.extensions.furthest_point_sampling import furthest_point_sampling
from .em_tools import EMTools


class GMMInitializer:
    # Capsules all the possible GMM-initialization methods for the EM algorithm(s)

    def __init__(self,
                 em_step_gaussians_subbatchsize: int = -1,
                 em_step_points_subbatchsize: int = -1,
                 dtype: torch.dtype = torch.float32,
                 eps: float = 1e-4):
        #   Creates a new GMM-initializer.
        #   em_step_gaussian_subbatchsize: int
        #       How many Gaussian Sub-Mixtures should be processed in the E- and M-Step at once (only relevant for
        #       initialization techniques randresp and fspmax)
        #       -1 means all Gaussians (default)
        #   em_step_points_subbatchsize: int
        #       How many points should be processed in the E- and M-Step at once (only relevant for
        #       initialization techniques randresp and fspmax)
        #       -1 means all Points (default)
        #   dtype: torch.dtype
        #       In which data type (precision) the operations should be performed. The final gmm is always
        #       converted to float32 though. Default: torch.float32
        #   eps: float
        #       Small value to be added to the Covariances for numerical stability. Default: 1e-4
        #
        self._em_step_gaussians_subbatchsize = em_step_gaussians_subbatchsize
        self._em_step_points_subbatchsize = em_step_points_subbatchsize
        self._dtype = dtype
        self._epsvar = eps

    def initialize_by_method_name(self, method_name: str, pcbatch: torch.Tensor, n_gaussians: int,
                                  n_sample_points: int = -1, weights: torch.Tensor = None):
        # Calls one of the initialization methods by its name
        # Parameters:
        #   method_name: str
        #       Name of the method to call. Options: 'randnormpos' ('rand1'), 'randresp' ('rand2'), 'fsp' ('adam1'),
        #       'fspmax' ('adam2'), 'kmeans-full', 'kmeans-fast' ('kmeans')
        #   pcbatch: torch.Tensor of size (batch_size, n_points, 3)
        #       Point cloud
        #   n_gaussians: int
        #       Number of Gaussians
        #   n_sample_points: int
        #       How many points to sample (if necessary), -1 = all points
        #   weights: torch.Tensor of size (batch_size, n_points) or None
        #       Additional weights for the points. Only used by randnormpos and kmeans-methods

        if method_name == 'randnormpos' or method_name == 'rand1':
            return self.initialize_randnormpos(pcbatch, n_gaussians, weights)
        elif method_name == 'randresp' or method_name == 'rand2':
            return self.initialize_randresp(pcbatch, n_gaussians, n_sample_points)
        elif method_name == 'fsp' or method_name == 'adam1':
            return self.initialize_fsp(pcbatch, n_gaussians)
        elif method_name == 'fspmax' or method_name == 'adam2':
            return self.initialize_fspmax(pcbatch, n_gaussians, n_sample_points)
        elif method_name == 'kmeans-full':
            return self.initialize_kmeans(pcbatch, n_gaussians, False, weights)
        elif method_name == 'kmeans-fast' or method_name == 'kmeans':
            return self.initialize_kmeans(pcbatch, n_gaussians, True, weights)
        else:
            raise Exception("Invalid Initialization Method for GMMInitializer")

    def initialize_randnormpos(self, pcbatch: torch.Tensor, n_gaussians: int, weights: torch.Tensor = None) \
            -> torch.Tensor:
        # Creates a new initial Gaussian Mixture (batch) for a given point cloud (batch).
        # The initialization is done according to McLachlan and Peel "Finite Mixture Models" (2000), Chapter 2.12.2
        # The positions are sampled from a normal distribution based on the empirical mean and covariances
        # of the point cloud. The covariances of the Gaussians are equal to the empirical covariances of the
        # point cloud.
        # Parameters:
        #   pcbatch: torch.Tensor(batch_size, n_points, 3)
        #   n_gaussians: int, Number of Gaussians
        #   weights: torch.Tensor(batch_size, n_points)
        #       Adds additional weights to the points (optional, may be None)
        batch_size = pcbatch.shape[0]
        point_count = pcbatch.shape[1]
        dtype = pcbatch.dtype

        if weights is None:
            # Calculate the mean pc position. shape: (bs, 1, 3)
            meanpos = pcbatch.mean(dim=1, keepdim=True)
            # Calcualte (point - meanpoint) pairs. Shape: (bs, np, 3, 1)
            diffs = (pcbatch - meanpos.expand(batch_size, point_count, 3)).unsqueeze(3)
            # Squeeze meanpos -> shape: (bs, 3)
            meanpos = meanpos.squeeze(1)
            # Calculate expected covariance. Shape: (bs, 3, 3)
            meancov = (diffs * diffs.transpose(-1, -2)).mean(dim=1)
        else:
            weights /= weights.sum(dim=1, keepdim=True)
            meanpos = (weights.unsqueeze(2) * pcbatch).sum(dim=1, keepdim=True)
            diffs = (pcbatch - meanpos.expand(batch_size, point_count, 3)).unsqueeze(3)
            meanpos = meanpos.squeeze(1)
            meancov = (weights.unsqueeze(2).unsqueeze(3) * (diffs * diffs.transpose(-1, -2))).sum(dim=1)
        # Calculated mean prior.
        meanweight = 1.0 / n_gaussians

        eps = (torch.eye(3, 3, dtype=dtype) * self._epsvar).view(1, 1, 1, 3, 3) \
            .expand(batch_size, 1, n_gaussians, 3, 3).cuda()

        # Sample positions from Gaussian -> shape: (bs, 1, ng, 3)
        positions = torch.zeros(batch_size, 1, n_gaussians, 3).to(dtype)
        for i in range(batch_size):
            positions[i, 0, :, :] = torch.tensor(
                np.random.multivariate_normal(meanpos[i, :].cpu(), meancov[i, :, :].cpu(), n_gaussians)).cuda()
        # Repeat covariances for each Gaussian -> shape: (bs, 1, ng, 3, 3)
        covariances = meancov.view(batch_size, 1, 1, 3, 3).expand(batch_size, 1, n_gaussians, 3, 3) + eps

        # Set weight for each Gaussian -> shape: (bs, 1, ng)
        weights = torch.zeros(batch_size, 1, n_gaussians).to(dtype)
        weights[:, :, :] = meanweight

        # pack gmm-mixture
        return gm.pack_mixture(weights.cuda(), positions.cuda(), covariances.cuda()).to(dtype)

    def initialize_randresp(self, pcbatch: torch.Tensor, n_gaussians: int, n_sample_points: int = -1) -> torch.Tensor:
        # Creates a new initial Gaussian Mixture (batch) for a given point cloud (batch).
        # The initialization is done according to McLachlan and Peel "Finite Mixture Models" (2000), Chapter 2.12.2
        # The responsibilities are created somewhat randomly and from these the M step calculates the Gaussians.
        # Parameters:
        #   pcbatch: torch.Tensor(batch_size, n_points, 3)
        #   n_gaussians: int (Number of Gaussians)
        #   n_sample_points: int (How many points to sample, -1 = all points)
        batch_size = pcbatch.shape[0]
        point_count = pcbatch.shape[1]
        dtype = pcbatch.dtype
        n_sample_points = min(point_count, n_sample_points)
        if n_sample_points == -1:
            n_sample_points = point_count
        if n_sample_points < point_count:
            sample_points = data_loading.sample(pcbatch, n_sample_points)
        else:
            sample_points = pcbatch

        eps = (torch.eye(3, 3, dtype=dtype) * self._epsvar).view(1, 1, 1, 3, 3) \
            .expand(batch_size, 1, n_gaussians, 3, 3).cuda()

        assignments = torch.randint(low=0, high=n_gaussians, size=(batch_size * n_sample_points,))
        point_indizes = torch.arange(0, n_sample_points).repeat(batch_size)
        batch_indizes = torch.arange(0, batch_size).repeat(n_sample_points, 1).transpose(-1, -2).reshape(-1)
        responsibilities = torch.zeros(batch_size, n_sample_points, n_gaussians).cuda()
        responsibilities[batch_indizes, point_indizes, assignments] = 1
        responsibilities = responsibilities.unsqueeze(1).to(self._dtype)

        gm_data = EMTools.TrainingData(batch_size, n_gaussians, self._dtype, eps)
        sp_rep = sample_points.unsqueeze(1).unsqueeze(3).expand(batch_size, 1, n_sample_points, n_gaussians, 3)
        EMTools.maximization(sp_rep, responsibilities, gm_data, torch.ones(batch_size, dtype=torch.bool),
                             eps, self._em_step_gaussians_subbatchsize, self._em_step_points_subbatchsize)
        return gm_data.pack_mixture_model().cuda().to(self._dtype)

    def initialize_fsp(self, pcbatch: torch.Tensor, n_gaussians: int):
        # Creates a new initial Gaussian Mixture (batch) for a given point cloud (batch).
        # The initialization is done according to Adam's method.
        # Furthest Point Sampling for mean selection, then assigning each point to the closest mean, then performing
        # an M step.
        # Parameters:
        #   pcbatch: torch.Tensor(batch_size, n_points, 3)
        #   n_gaussians: int (number of Gaussians)
        batch_size = pcbatch.shape[0]

        eps = (torch.eye(3, 3, dtype=self._dtype) * self._epsvar).view(1, 1, 1, 3, 3) \
            .expand(batch_size, 1, n_gaussians, 3, 3).cuda()

        sampled = furthest_point_sampling.apply(pcbatch.float(), n_gaussians).to(torch.long).reshape(-1)
        batch_indizes = torch.arange(0, batch_size).repeat(n_gaussians, 1).transpose(-1, -2).reshape(-1)
        gmpositions = pcbatch[batch_indizes, sampled, :].view(batch_size, 1, n_gaussians, 3)
        gmcovariances = torch.zeros(batch_size, 1, n_gaussians, 3, 3).to(self._dtype).cuda() + eps
        gmcovariances[:, :, :] = torch.eye(3, dtype=self._dtype).cuda()
        gmpriors = torch.zeros(batch_size, 1, n_gaussians).to(self._dtype).cuda()
        gmpriors[:, :, :] = 1 / n_gaussians

        return gm.pack_mixture(gmpriors, gmpositions, gmcovariances)

    def initialize_fspmax(self, pcbatch: torch.Tensor, n_gaussians: int, n_sample_points: int = -1):
        # Creates a new initial Gaussian Mixture (batch) for a given point cloud (batch).
        # The initialization is done according to Adam's method.
        # Furthest Point Sampling for mean selection, then assigning each point to the closest mean, then performing
        # an M step
        # Parameters:
        #   pcbatch: torch.Tensor(batch_size, n_points, 3)
        #   n_gaussians: int (number of Gaussians)
        #   n_sample_points: int (How many points to sample, -1 = all points)
        batch_size = pcbatch.shape[0]
        point_count = pcbatch.shape[1]

        n_sample_points = min(point_count, n_sample_points)
        if n_sample_points == -1:
            n_sample_points = point_count
        if n_sample_points < point_count:
            sample_points = data_loading.sample(pcbatch, n_sample_points)
        else:
            sample_points = pcbatch

        eps = (torch.eye(3, 3, dtype=self._dtype) * self._epsvar).view(1, 1, 1, 3, 3) \
            .expand(batch_size, 1, n_gaussians, 3, 3).cuda()

        mix = self.initialize_fsp(pcbatch, n_gaussians)

        gm_data = EMTools.TrainingData(batch_size, n_gaussians, self._dtype, eps)
        running = torch.ones(batch_size, dtype=torch.bool)
        gm_data.set_positions(gm.positions(mix), running)
        gm_data.set_covariances(gm.covariances(mix), running)
        gm_data.set_priors(gm.weights(mix), running)

        sp_rep = sample_points.unsqueeze(1).unsqueeze(3).expand(batch_size, 1, n_sample_points, n_gaussians, 3)

        responsibilities, llh = EMTools.expectation(sp_rep, gm_data, n_gaussians, running,
                                                    self._em_step_gaussians_subbatchsize,
                                                    self._em_step_points_subbatchsize)

        # resp dimension: (batch_size, 1, n_points, n_gaussians)
        # let's find the maximum per gaussian
        assignments = responsibilities.argmax(dim=3).view(-1)
        point_indizes = torch.arange(0, n_sample_points).repeat(batch_size)
        batch_indizes = torch.arange(0, batch_size).repeat(n_sample_points, 1).transpose(-1, -2).reshape(-1)
        assignedresps = torch.zeros(batch_size, n_sample_points, n_gaussians).cuda()
        assignedresps[batch_indizes, point_indizes, assignments] = 1
        assignedresps = assignedresps.unsqueeze(1).to(self._dtype)

        EMTools.maximization(sp_rep, assignedresps, gm_data, running, eps, self._em_step_gaussians_subbatchsize,
                             self._em_step_points_subbatchsize)
        return gm_data.pack_mixture_model().cuda().to(self._dtype)

    def initialize_kmeans(self, pcbatch: torch.Tensor, n_gaussians: int, fast: bool = True, weights: torch.Tensor = None) -> torch.Tensor:
        # Creates a new initial Gaussian Mixture (batch) for a given point cloud (batch).
        # Calculates the initial Gaussian positions using KMeans.
        # Parameters:
        #   pcbatch: torch.Tensor(batch_size, n_points, 3)
        #   n_gaussians: int (number of Gaussians)
        #   fast: bool
        #       The fast version only performs a few iterations rather than the default scikit-options
        #   weights: torch.Tensor of size (batch_size, n_points)
        #       Adds additional weights to the points (optional, may be None)
        batch_size = pcbatch.shape[0]
        point_count = pcbatch.shape[1]

        positions = torch.zeros(batch_size, 1, n_gaussians, 3, dtype=self._dtype).cuda()
        covariances = torch.zeros(batch_size, 1, n_gaussians, 3, 3, dtype=self._dtype).cuda()
        priors = torch.zeros(batch_size, 1, n_gaussians, dtype=self._dtype).cuda()

        eps = (torch.eye(3, 3, dtype=self._dtype) * self._epsvar).view(1, 1, 1, 3, 3) \
            .expand(batch_size, 1, n_gaussians, 3, 3).cuda()

        if weights is not None:
            weights /= weights.sum(dim=1, keepdim=True)

        for batch in range(batch_size):
            if fast:
                km = KMeans(n_gaussians, n_init=1, max_iter=20).fit(pcbatch[batch].cpu(), weights[batch].cpu() if weights is not None else None)
            else:
                km = KMeans(n_gaussians).fit(pcbatch[batch].cpu(), weights[batch].cpu() if weights is not None else None)
            positions[batch, 0] = torch.tensor(km.cluster_centers_).cuda()
            labels = torch.tensor(km.labels_).cuda()
            allidcs = torch.tensor(range(n_gaussians)).cuda()
            # mask: (ng, np) True where there is an assignment
            mask = labels.eq(allidcs.view(-1, 1))
            count_per_cluster = mask.sum(1)
            # points_rep: (ng, np, 3)
            points_rep = pcbatch[batch].unsqueeze(0).repeat(n_gaussians,1,1)
            points_rep = (points_rep - positions[batch, 0].unsqueeze(1).expand(n_gaussians, point_count, 3))\
                .unsqueeze(3)
            points_rep[~mask] = 0
            if weights is None:
                covariances[batch, 0] = (points_rep * points_rep.transpose(-1, -2)).sum(1) \
                    / count_per_cluster.view(-1, 1, 1) + eps
            else:
                weights_mask = mask * weights[batch].unsqueeze(0).expand(n_gaussians, point_count)
                weight_sum = weights_mask.sum(1).view(-1, 1, 1)
                covariances[batch, 0] = (weights_mask.unsqueeze(2).unsqueeze(3) * (points_rep * points_rep.transpose(-1, -2))).sum(1) / weight_sum + eps
            priors[batch, 0, :] = 1 / n_gaussians

        return gm.pack_mixture(priors, positions, covariances)