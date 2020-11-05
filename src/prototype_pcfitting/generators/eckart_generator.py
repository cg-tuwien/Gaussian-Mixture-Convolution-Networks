from prototype_pcfitting import GMMGenerator, GMLogger, data_loading, Scaler, ScalingMethod
from prototype_pcfitting import TerminationCriterion, MaxIterationTerminationCriterion
from gmc.cpp.extensions.furthest_point_sampling import furthest_point_sampling
import torch
import gmc.mixture as gm
import math


class EckartGenerator(GMMGenerator):
    # GMM Generator using Expectation Sparsification by Eckart

    def __init__(self,
                 n_gaussians_per_node: int,
                 n_levels: int,
                 dtype: torch.dtype = torch.float32):
        self._n_gaussians_per_node = n_gaussians_per_node
        self._n_levels = n_levels
        self._dtype = dtype

    def set_logging(self, logger: GMLogger = None):
        # Sets logging options
        # Paramters:
        #   logger: GMLogger
        #       GMLogger object to call every iteration
        #
        self._logger = logger

    def generate(self, pcbatch: torch.Tensor, gmbatch: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        assert (gmbatch is None), "EckartGenerator cannot improve existing GMMs"

        batch_size = pcbatch.shape[0]

        assert (batch_size is 1), "EckartGenerator currently does not support batchsizes > 1"
        point_count = pcbatch.shape[1]
        pcbatch = pcbatch.to(self._dtype).cuda()

        self._eps = (torch.eye(3, 3, dtype=self._dtype) * 1e-6).view(1, 1, 1, 3, 3) \
            .expand(batch_size, 1, self._n_gaussians_per_node, 3, 3).cuda()

        parents = torch.zeros(1, point_count).to(torch.long).cuda()
        parents[:, :] = -1

        hierarchy = []
        
        gmmindex = -2

        # Iterate over levels
        for l in range(self._n_levels):
            # Iterate over GMMs for this level
            for j in range(self._n_gaussians_per_node ** l):   # <- WE SHOULD GET RID OF THIS LOOP! PARALLEL!
                print("Level ", l, " Gaussian ", j)

                # Calculate index of this GMM
                gmmindex += 1

                # Find all points that have this GMM as their parent
                # This will be hard to parallelize, as the point counts might be different per batch entry
                # currently only supports batch_size == 1
                point_indizes = torch.nonzero(parents == gmmindex)
                relevant_points = pcbatch[:, point_indizes[:, 1], :]
                relevant_point_count = point_indizes.shape[0]

                # Relevant point count might be zero
                if relevant_point_count == 0:
                    gm_data = self._initialize_gm_on_unit_cube(batch_size)
                    gm_data.multiply_weights(0)
                    hierarchy.append(gm_data)
                    continue

                # Scale them to the unit cube
                scaler = Scaler(ScalingMethod.LARGEST_TO_ONE)
                scaler.set_pointcloud_batch(relevant_points)
                relevant_points_scaled = scaler.scale_down_pc(relevant_points)

                # Create initial GMM
                gm_data = self._initialize_gm_on_unit_cube(batch_size)

                # Iterate E- and M-steps
                # Calculate the new parent indizes for these points

                iteration = 0

                while True:
                    iteration += 1

                    points_rep = relevant_points.unsqueeze(1).unsqueeze(3).expand(batch_size, 1, relevant_point_count, self._n_gaussians_per_node, 3)

                    responsibilities, losses = self._expectation(points_rep, gm_data)

                    loss = losses.sum()
                    assert not torch.isnan(loss).any()

                    if iteration == 20:
                        current_parent_indizes = responsibilities.argmax(dim=3)
                        current_parent_indizes += (gmmindex+1)*self._n_gaussians_per_node
                        parents[:, point_indizes[:, 1]] = current_parent_indizes
                        break

                    self._maximization(points_rep, responsibilities, gm_data)

                gm_data.scale_up(scaler)
                hierarchy.append(gm_data)
        res_gm, res_gmm = self._construct_gm_from_hierarchy(hierarchy)
        res_gm = res_gm.float()
        res_gmm = res_gmm.float()

        return res_gm, res_gmm

    def _construct_gm_from_hierarchy(self, hierarchy) -> (torch.Tensor, torch.Tensor):
        last_h_n = self._n_gaussians_per_node ** (self._n_levels - 1)
        n = self._n_gaussians_per_node ** self._n_levels
        gm = torch.zeros(1, 1, n, 13).to(self._dtype).cuda()
        gmm = torch.zeros(1, 1, n, 13).to(self._dtype).cuda()
        for j in range(last_h_n):  # this should also be possible in parallel
            gm_index = len(hierarchy) - 1 - j
            gm_data = hierarchy[gm_index]
            for l in range(self._n_levels - 1):
                parent_index = math.floor(gm_index / self._n_gaussians_per_node)
                parents_child_index = gm_index % self._n_gaussians_per_node
                factor = hierarchy[parent_index].get_priors()[0, 0, parents_child_index]
                gm_data.multiply_weights(factor)

            mix = gm_data.pack_mixture()
            mix_mod = gm_data.pack_mixture_model()
            startidx = j*self._n_gaussians_per_node
            gm[:, :, startidx:startidx + self._n_gaussians_per_node] = mix
            gmm[:, :, startidx:startidx + self._n_gaussians_per_node] = mix_mod
        return gm, gmm


    def _expectation(self, points_rep: torch.Tensor, gm_data) -> (torch.Tensor, torch.Tensor):
        # This performs the Expectation step of the EM Algorithm. This calculates 1) the responsibilities.
        # So the probabilities, how likely each point belongs to each gaussian and 2) the overall Log-Likelihood
        # of this GM given the point cloud.
        # The calculations are performed numerically stable in Log-Space!
        # Parameters:
        #   points_rep: torch.Tensor of shape (batch_size, 1, n_points, n_gaussians, 3)
        #       This is a expansion of the (sampled) point cloud, repeated n_gaussian times along dimension 4
        #   gm_data: TrainingData
        #       The current GM-object
        # Returns:
        #   responsibilities: torch.Tensor of shape (batch_size, 1, n_points, n_gaussians)
        #   losses: torch.Tensor of shape (batch_size): Negative Log-Likelihood for each GM

        batch_size = points_rep.shape[0]
        n_sample_points = points_rep.shape[2]

        # This uses the fact that
        # log(a * exp(-0.5 * M(x))) = log(a) + log(exp(-0.5 * M(x))) = log(a) - 0.5 * M(x)

        # GM-Positions, expanded for each PC point. shape: (bs, 1, np, ng, 3)
        gmpositions_rep = gm_data.get_positions() \
            .unsqueeze(2).expand(batch_size, 1, n_sample_points, self._n_gaussians_per_node, 3)
        # GM-Inverse Covariances, expanded for each PC point. shape: (bs, 1, np, ng, 3, 3)
        gmicovs_rep = gm_data.get_inversed_covariances() \
            .unsqueeze(2).expand(batch_size, 1, n_sample_points, self._n_gaussians_per_node, 3, 3)
        # Tensor of {PC-point minus GM-position}-vectors. shape: (bs, 1, np, ng, 3, 1)
        grelpos = (points_rep - gmpositions_rep).unsqueeze(5)
        # Tensor of 0.5 times the Mahalanobis distances of PC points to Gaussians. shape: (bs, 1, np, ng)
        expvalues = 0.5 * \
            torch.matmul(grelpos.transpose(-2, -1), torch.matmul(gmicovs_rep, grelpos)).squeeze(5).squeeze(4)
        # Logarithmized GM-Priors, expanded for each PC point. shape: (bs, 1, np, ng)
        gmpriors_log_rep = \
            torch.log(gm_data.get_amplitudes().unsqueeze(2).expand(batch_size, 1, n_sample_points, self._n_gaussians_per_node))
        # The logarithmized likelihoods of each point for each gaussian. shape: (bs, 1, np, ng)
        likelihood_log = gmpriors_log_rep - expvalues
        # Logarithmized Likelihood for each point given the GM. shape: (bs, 1, np, ng)
        llh_sum = torch.logsumexp(likelihood_log, dim=3, keepdim=True)
        # Logarithmized Mean Likelihood for all points. shape: (bs)
        losses = -llh_sum.mean(dim=2).view(batch_size)
        # Calculating responsibilities and returning them and the mean loglikelihoods
        return torch.exp(likelihood_log - llh_sum), losses

    def _maximization(self, points_rep: torch.Tensor, responsibilities: torch.Tensor, gm_data):
        # This performs the Maximization step of the EM Algorithm.
        # Updates the GM-Model given the responsibilities which resulted from the E-Step.
        # Parameters:
        #   points_rep: torch.Tensor of shape (batch_size, 1, n_points, n_gaussians, 3)
        #       This is a expansion of the (sampled) point cloud, repeated n_gaussian times along dimension 4
        #   responsibilities: torch.Tensor of shape (batch_size, 1, n_points, n_gaussians)
        #       This is the result of the E-step.
        #   gm_data: TrainingData
        #       The current GM-object (will be changed)

        batch_size = points_rep.shape[0]
        n_sample_points = points_rep.shape[2]

        # Calculate n_k (the amount of points assigned to each Gaussian). shape: (bs, 1, ng)
        n_k = responsibilities.sum(dim=2)

        # Calculate new GM positions with the formula \sum_{n=1}^{N}{r_{nk} * x_n} / n_k
        # Multiply responsibilities and points. shape: (bs, 1, np, ng, 3)
        multiplied = responsibilities.unsqueeze(4).expand_as(points_rep) * points_rep
        # New Positions -> Build sum and divide by n_k. shape: (bs, 1, 1, ng, 3)
        new_positions = multiplied.sum(dim=2, keepdim=True) / n_k.unsqueeze(2).unsqueeze(4)
        # Repeat positions for each point, for later calculation. shape: (bs, 1, np, ng, 3)
        new_positions_rep = new_positions.expand(batch_size, 1, n_sample_points, self._n_gaussians_per_node, 3)
        # Squeeze positions for result. shape: (bs, 1, ng, 3)
        new_positions = new_positions.squeeze(2)

        # Calculate new GM covariances with the formula \sum_{n=1}^N{r_{nk}*(x_n-\mu_k)(x_n-\mu_k)^T} / n_k + eps
        # Tensor of (x_n-\mu_k)-vectors. shape: (bs, 1, np, ng, 3, 1)
        relpos = (points_rep - new_positions_rep).unsqueeze(5)
        # Tensor of r_{nk}*(x_n-\mu_k)(x_n-\mu_k)^T} matrices. shape: (bs, 1, np, ng, 3, 3)
        matrix = (relpos * (relpos.transpose(-1, -2))) * responsibilities.unsqueeze(4).unsqueeze(5)
        # New Covariances -> Sum matrices, divide by n_k and add eps. shape: (bs, 1, ng, 3, 3)
        new_covariances = matrix.sum(dim=2) / n_k.unsqueeze(3).unsqueeze(4) + self._eps

        # Calculate new GM priors with the formula N_k / N. shape: (bs, 1, ng)
        new_priors = n_k / n_sample_points

        # Handling of invalid Gaussians! If all responsibilities of a Gaussian are zero, the previous code will
        # set the prior of it to zero and the covariances and positions to NaN
        # To avoid NaNs, we will then replace those invalid values with 0 (pos) and eps (cov).
        new_positions[new_priors == 0] = torch.tensor([0.0, 0.0, 0.0], dtype=self._dtype).cuda()
        new_covariances[new_priors == 0] = self._eps[0, 0, 0, :, :]

        # Update GMData
        gm_data.set_positions(new_positions)
        gm_data.set_covariances(new_covariances)
        gm_data.set_priors(new_priors)

    def _initialize_gm_on_unit_cube(self, batch_size):
        position_templates = torch.tensor([
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 1]
        ], dtype=self._dtype).cuda()
        if self._n_gaussians_per_node <= 8:
            gmpositions = position_templates[0:self._n_gaussians_per_node].unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        else:
            gmpositions = position_templates.unsqueeze(0).unsqueeze(0).\
                repeat(batch_size, 1, math.ceil(self._n_gaussians_per_node / 8), 1)
            gmpositions = gmpositions[:, 0:self._n_gaussians_per_node, :]
        gmcovs = torch.eye(3).to(self._dtype).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(0).\
            repeat(batch_size, 1, self._n_gaussians_per_node, 1, 1)
        gmweights = torch.zeros(batch_size, 1, self._n_gaussians_per_node).to(self._dtype).cuda()
        gmweights[:, :, :] = 1 / self._n_gaussians_per_node

        gmdata = self.GMTrainingData()
        gmdata.set_positions(gmpositions)
        gmdata.set_covariances(gmcovs)
        gmdata.set_priors(gmweights)
        return gmdata

    class GMTrainingData:
        # Helper class. Capsules all relevant training data of the current GM batch.
        # positions, covariances and priors are stored as-is and can be set.
        # inversed covariances are calcualted whenever covariances are set.
        # amplitudes are calculated from priors (or vice versa).
        # Note that priors or amplitudes should always be set after the covariances are set,
        # otherwise the conversion is not correct anymore.

        def __init__(self):
            self._positions = None
            self._amplitudes = None
            self._priors = None
            self._covariances = None
            self._inversed_covariances = None

        def set_positions(self, positions):
            self._positions = positions

        def set_covariances(self, covariances):
            self._covariances = covariances
            self._inversed_covariances = covariances.inverse().contiguous()

        def set_amplitudes(self, amplitudes):
            self._amplitudes = amplitudes
            self._priors = amplitudes * (self._covariances.det().sqrt() * 15.74960995)

        def set_priors(self, priors):
            self._priors = priors
            self._amplitudes = priors / (self._covariances.det().sqrt() * 15.74960995)

        def get_positions(self):
            return self._positions

        def get_covariances(self):
            return self._covariances

        def get_inversed_covariances(self):
            return self._inversed_covariances

        def get_priors(self):
            return self._priors

        def get_amplitudes(self):
            return self._amplitudes

        def pack_mixture(self):
            return gm.pack_mixture(self._amplitudes, self._positions, self._covariances)

        def pack_mixture_model(self):
            return gm.pack_mixture(self._priors, self._positions, self._covariances)

        def multiply_weights(self, factor):
            self._priors *= factor
            self._amplitudes *= factor

        def scale_up(self, scaler: Scaler):
            newpriors, newpositions, newcovariances = \
                scaler.scale_up_gmm_wpc(self._priors, self._positions, self._covariances)
            self.set_positions(newpositions)
            self.set_covariances(newcovariances)
            self.set_priors(newpriors)
