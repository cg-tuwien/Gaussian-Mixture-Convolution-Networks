from prototype_pcfitting import GMMGenerator, GMLogger, data_loading, Scaler, ScalingMethod
from prototype_pcfitting import TerminationCriterion, MaxIterationTerminationCriterion
from gmc.cpp.extensions.furthest_point_sampling import furthest_point_sampling
from .level_scaler2 import LevelScaler2
import torch
import gmc.mixture as gm
import math


class EckartGenerator3(GMMGenerator):
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

        self._eps = (torch.eye(3, 3, dtype=self._dtype) * 1e-6).view(1, 1, 1, 3, 3).cuda()

        parent_per_point = torch.zeros(1, point_count).to(torch.long).cuda()
        parent_per_point[:, :] = 0

        hierarchy = []

        for l in range(self._n_levels):
            parentcount_for_level = self._n_gaussians_per_node ** l
            gausscount_for_level = self._n_gaussians_per_node ** (l+1)
            relevant_parents = torch.arange(0, parentcount_for_level).cuda() # (gausscount)

            scaler = LevelScaler2()
            scaler.set_pointcloud(pcbatch, parent_per_point, parentcount_for_level)
            points_scaled = scaler.scale_down_pc(pcbatch)       # (1, np, 3)

            gm_data = self._initialize_gms_on_unit_cube(relevant_parents)

            iteration = 0
            while True:
                iteration += 1

                points = points_scaled.unsqueeze(1).unsqueeze(3) # (1, 1, np, 1, 3)

                responsibilities, losses = self._expectation(points, gm_data, parent_per_point, relevant_parents)
                loss = losses.sum()
                print("Iteration ", iteration, ". Summed Loss (NOT RELIEABLE): ", loss)

                if iteration == 20:
                    # update parent_per_point for next level from responsibilities
                    new_parent_indices = responsibilities.argmax(dim=3)
                    parent_per_point = new_parent_indices.view(1, -1)
                    break

                self._maximization(points, responsibilities, gm_data)

            gm_data.scale_up(scaler)
            # add gm to hierarchy
            hierarchy.append(gm_data)

        res_gm, res_gmm = self._construct_gm_from_hierarchy(hierarchy)
        res_gm = res_gm.float()
        res_gmm = res_gmm.float()

        return res_gm, res_gmm

    def _construct_gm_from_hierarchy(self, hierarchy) -> (torch.Tensor, torch.Tensor):
        last_gm = hierarchy[-1]
        for g in range(len(last_gm)):
            multpl = 1
            gidx = last_gm.get_parents()[0, 0, g]
            for l in range(self._n_levels-2, -1, -1):
                gm_priors = hierarchy[l].get_priors()
                multpl *= gm_priors[0, 0, gidx]
                gm_parents = hierarchy[l].get_parents()
                gidx = gm_parents[0, 0, gidx]
            last_gm.get_priors()[0, 0, g] *= multpl
            last_gm.get_amplitudes()[0, 0, g] *= multpl
        last_gm.remove_invalids()
        return last_gm.pack_mixture(), last_gm.pack_mixture_model()

    def _initialize_gms_on_unit_cube(self, relevant_parents):
        gmcount = relevant_parents.shape[0]
        gmparents = relevant_parents.repeat(1, self._n_gaussians_per_node).view(self._n_gaussians_per_node, -1).transpose(-1, -2).reshape(1, 1, -1)
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
            gmpositions = position_templates[0:self._n_gaussians_per_node].unsqueeze(0).unsqueeze(0).repeat(1, 1,
                                                                                                            gmcount, 1)
        else:
            gmpositions = position_templates.unsqueeze(0).unsqueeze(0). \
                repeat(1, 1, math.ceil(self._n_gaussians_per_node / 8), 1)
            gmpositions = gmpositions[:, :, 0:self._n_gaussians_per_node, :].repeat(1, 1, gmcount, 1)
        gmcovs = torch.eye(3).to(self._dtype).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(0). \
            repeat(1, 1, self._n_gaussians_per_node * gmcount, 1, 1)
        gmweights = torch.zeros(1, 1, self._n_gaussians_per_node * gmcount).to(self._dtype).cuda()
        gmweights[:, :, :] = 1 / self._n_gaussians_per_node

        gmdata = self.GMTrainingData()
        gmdata.set_positions(gmpositions)
        gmdata.set_covariances(gmcovs)
        gmdata.set_priors(gmweights)
        gmdata.set_parents(gmparents)
        return gmdata

    def _expectation(self, points: torch.Tensor, gm_data, parent_per_point: torch.Tensor, relevant_parents: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # This performs the Expectation step of the EM Algorithm. This calculates 1) the responsibilities.
        # So the probabilities, how likely each point belongs to each gaussian and 2) the overall Log-Likelihood
        # of this GM given the point cloud.
        # The calculations are performed numerically stable in Log-Space!
        # Parameters:
        #   points_rep: torch.Tensor of shape (batch_size, 1, n_points, n_gaussians_per_node, 3)
        #       This is a expansion of the (sampled) point cloud, repeated n_gaussian times along dimension 4
        #   gm_data: TrainingData
        #       The current GM-object
        # Returns:
        #   responsibilities: torch.Tensor of shape (batch_size, 1, n_points, n_gaussians)
        #   losses: torch.Tensor of shape (batch_size): Negative Log-Likelihood for each GM

        batch_size = points.shape[0]
        n_sample_points = points.shape[2]
        points_rep = points.expand(batch_size, 1, n_sample_points, self._n_gaussians_per_node, 3)
        all_gauss_count = gm_data.get_positions().shape[2]
        parent_count = relevant_parents.shape[0]

        # relevant_parents.indizes_of(parent_per_point) -> mapping to [0,m-1] (1,np)
        # parent_idx_per_point = torch.nonzero(parent_per_point.view(1, -1, 1) == relevant_parents)[:, 2].unsqueeze(0)

        # mask_indizes is a list of indizes of a) points with their corresponding b) gauss (child) indizes
        mask_indizes = torch.zeros(n_sample_points, 2, dtype=torch.long).cuda()
        mask_indizes[:, 0] = torch.arange(0, n_sample_points, dtype=torch.long).cuda()
        mask_indizes[:, 1] = parent_per_point.view(-1)
        mask_indizes = mask_indizes.repeat(1, 1, self._n_gaussians_per_node)
        mask_indizes[:, :, 1::2] *= self._n_gaussians_per_node # torch.ones(self._n_gaussians_per_node, dtype=torch.long).cuda()
        mask_indizes[:, :, 1::2] += torch.arange(0, self._n_gaussians_per_node, dtype=torch.long).cuda()
        mask_indizes = mask_indizes.view(n_sample_points * self._n_gaussians_per_node, 2)
        # TODO: maybe its much better if we initially work with the indicator matrix. it should be easy to
        # calcualte mask_indizes from that using nonzero. we might even be able to use it in initialization

        # This uses the fact that
        # log(a * exp(-0.5 * M(x))) = log(a) + log(exp(-0.5 * M(x))) = log(a) - 0.5 * M(x)

        # GM-Positions, expanded for each PC point. shape: (bs, 1, np, ng, 3)
        gmpositions_rep = gm_data.get_positions() \
            .unsqueeze(2).expand(batch_size, 1, n_sample_points, all_gauss_count, 3)
        gmpositions_rep = gmpositions_rep[:, :, mask_indizes[:,0], mask_indizes[:, 1], :].view(batch_size, 1, n_sample_points, self._n_gaussians_per_node, 3)
        # GM-Inverse Covariances, expanded for each PC point. shape: (bs, 1, np, ng, 3, 3)
        gmicovs_rep = gm_data.get_inversed_covariances() \
            .unsqueeze(2).expand(batch_size, 1, n_sample_points, all_gauss_count, 3, 3)
        gmicovs_rep = gmicovs_rep[:, :, mask_indizes[:, 0], mask_indizes[:, 1], :, :].view(batch_size, 1, n_sample_points, self._n_gaussians_per_node, 3, 3)
        # Tensor of {PC-point minus GM-position}-vectors. shape: (bs, 1, np, ng, 3, 1)
        grelpos = (points_rep - gmpositions_rep).unsqueeze(5)
        # Tensor of 0.5 times the Mahalanobis distances of PC points to Gaussians. shape: (bs, 1, np, ng)
        expvalues = 0.5 * \
                    torch.matmul(grelpos.transpose(-2, -1), torch.matmul(gmicovs_rep, grelpos)).squeeze(5).squeeze(4)
        # Logarithmized GM-Priors, expanded for each PC point. shape: (bs, 1, np, ng)
        gmpriors_log_rep = \
            torch.log(gm_data.get_amplitudes().unsqueeze(2).expand(batch_size, 1, n_sample_points, all_gauss_count))
        gmpriors_log_rep = gmpriors_log_rep[:, :, mask_indizes[:, 0], mask_indizes[:, 1]].view(batch_size, 1, n_sample_points, self._n_gaussians_per_node)
        # The logarithmized likelihoods of each point for each gaussian. shape: (bs, 1, np, ng)
        likelihood_log = gmpriors_log_rep - expvalues
        # Logarithmized Likelihood for each point given the GM. shape: (bs, 1, np, 1)
        llh_sum = torch.logsumexp(likelihood_log, dim=3, keepdim=True)

        # Calculate Loss per GM (per Parent)
        # das passt nicht, denn wir haben den falschen gausscount. wir wollens pro parent
        indicator = parent_per_point.unsqueeze(2).repeat(1, 1, parent_count)    # (1, np, nP)
        indicator = (indicator == (torch.arange(0, parent_count).cuda().view(1, 1, -1).repeat(1, n_sample_points, 1)))
        losses = (indicator * llh_sum.squeeze(1).repeat(1, 1, parent_count)).sum(dim=1) / indicator.sum(dim=1) # (1, ng)

        # assert not torch.isnan(losses).any()
        losses[torch.isnan(losses)] = 0 # temporary fix: as there are parents without gaussians there will be no loss defined for some

        # responsibilities[indicator == 1] = [1, 2, 3]...
        responsibilities_local = torch.exp(likelihood_log - llh_sum) # (bs, 1, np, ngLocal)
        responsibilities_global = torch.zeros(1, 1, n_sample_points, all_gauss_count, dtype=self._dtype).cuda()
        responsibilities_global[:, :, mask_indizes[:,0], mask_indizes[:,1]] = responsibilities_local.view(1,1,-1)

        # Calculating responsibilities and returning them and the mean loglikelihoods
        return responsibilities_global, losses

    def _maximization(self, points: torch.Tensor, responsibilities: torch.Tensor, gm_data):
        # This performs the Maximization step of the EM Algorithm.
        # Updates the GM-Model given the responsibilities which resulted from the E-Step.
        # Parameters:
        #   points_rep: torch.Tensor of shape (batch_size, 1, n_points, n_gaussians, 3)
        #       This is a expansion of the (sampled) point cloud, repeated n_gaussian times along dimension 4
        #   responsibilities: torch.Tensor of shape (batch_size, 1, n_points, n_gaussians)
        #       This is the result of the E-step.
        #   gm_data: TrainingData
        #       The current GM-object (will be changed)
        #   indicator: torch.Tensor of shape (bs(1), 1, np, ng)

        batch_size = points.shape[0]
        n_sample_points = points.shape[2]
        all_gauss_count = responsibilities.shape[3]
        points_rep = points.expand(1, 1, n_sample_points, all_gauss_count, 3)

        # ToDo: Out of Memory. Optimize or Sample! For now we will reduce the point count

        T_0 = responsibilities.sum(dim=2)   # shape: (1, 1, J) J = G_GLOBAL
        T_1 = (points_rep * responsibilities.unsqueeze(4)).sum(dim=2)    # shape: (1, 1, J, 3)
        T_2 = ((points_rep.unsqueeze(5) * points_rep.unsqueeze(5).transpose(-1, -2))
               * responsibilities.unsqueeze(4).unsqueeze(5)).sum(dim = 2)       # shape: (1, 1, J, 3, 3)

        # ToDo: This more simple calculation can also be used for normal EM

        new_positions = T_1 / T_0.unsqueeze(3)    # (1, 1, J, 3)
        new_covariances = T_2 / T_0.unsqueeze(3).unsqueeze(4) - \
                          (new_positions.unsqueeze(4) * new_positions.unsqueeze(4).transpose(-1, -2)) \
                          + self._eps.expand_as(T_2)
        new_priors = T_0 / T_0.sum()

        # Handling of invalid Gaussians! If all responsibilities of a Gaussian are zero, the previous code will
        # set the prior of it to zero and the covariances and positions to NaN
        # To avoid NaNs, we will then replace those invalid values with 0 (pos) and eps (cov).
        new_positions[T_0 == 0] = torch.tensor([0.0, 0.0, 0.0], dtype=self._dtype).cuda()
        new_covariances[T_0 == 0] = self._eps[0, 0, 0, :, :]
        new_priors[T_0 == 0] = 0

        # Update GMData
        gm_data.set_positions(new_positions)
        gm_data.set_covariances(new_covariances)
        gm_data.set_priors(new_priors)

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
            self._parents = None  # (1, 1, g) Indizes of parent Gaussians on parent Level

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

        def set_parents(self, parents):
            self._parents = parents

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

        def get_parents(self):
            return self._parents

        def pack_mixture(self):
            return gm.pack_mixture(self._amplitudes, self._positions, self._covariances)

        def pack_mixture_model(self):
            return gm.pack_mixture(self._priors, self._positions, self._covariances)

        def multiply_weights(self, factor):
            self._priors *= factor
            self._amplitudes *= factor

        def scale_up(self, scaler: LevelScaler2):
            newpriors, newpositions, newcovariances = \
                scaler.scale_up_gmm_wpc(self._priors, self._positions, self._covariances)
            self.set_positions(newpositions)
            self.set_covariances(newcovariances)
            self.set_priors(newpriors)

        def remove_invalids(self):
            valids = self._priors > 0
            self._positions = self._positions[valids].view(1, 1, -1, 3)
            self._amplitudes = self._amplitudes[valids].view(1, 1, -1)
            self._priors = self._priors[valids].view(1, 1, -1)
            self._covariances = self._covariances[valids].view(1, 1, -1, 3, 3)
            self._inversed_covariances = self._inversed_covariances[valids].view(1, 1, -1, 3, 3)
            self._parents = self._parents[valids].view(1, 1, -1)

        def __len__(self):
            return self._priors.shape[2]
