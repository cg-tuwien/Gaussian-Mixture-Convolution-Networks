from prototype_pcfitting import GMMGenerator, GMLogger
from prototype_pcfitting import TerminationCriterion, MaxIterationTerminationCriterion
from prototype_pcfitting.error_functions import LikelihoodLoss
from .level_scaler import LevelScaler
import torch
import gmc.mixture as gm
import math


class EckartGeneratorSP(GMMGenerator):
    # GMM Generator using Expectation Sparsification with soft partitioning by Eckart

    def __init__(self,
                 n_gaussians_per_node: int,
                 n_levels: int,
                 partition_treshold: float,
                 termination_criterion: TerminationCriterion = MaxIterationTerminationCriterion(20),
                 m_step_gaussians_subbatchsize: int = -1,
                 m_step_points_subbatchsize: int = -1,
                 dtype: torch.dtype = torch.float32):
        # Constructor. Creates a new EckartGenerator.
        # Parameters:
        #   n_gaussians_per_node: int
        #       To how many Gaussians each Gaussian should be expanded to in the next level
        #   n_levels: int
        #       Number of levels
        #   termination_criterion: TerminationCriterion
        #       Defining when to terminate PER LEVEL
        #   m_step_gaussian_subbatchsize: int
        #       How many Gaussian Sub-Mixtures should be processed in the M-Step at once (see _maximization)
        #       -1 means all Gaussians (default)
        #   m_step_points_subbatchsize: int
        #       How many points should be processed in the M-Step at once (see _maximization)
        #       -1 means all Points (default)
        #   dtype: torch.dtype
        #       In which data type (precision) the oepration should be performed. The final gmm is always
        #       converted to float32 though. Default: float32
        self._n_gaussians_per_node = n_gaussians_per_node
        self._n_levels = n_levels
        self._partition_threshold = partition_treshold
        self._termination_criterion = termination_criterion
        self._m_step_gaussians_subbatchsize = m_step_gaussians_subbatchsize
        self._m_step_points_subbatchsize = m_step_points_subbatchsize
        self._dtype = dtype
        self._logger = None
        self._eps = None

    def set_logging(self, logger: GMLogger = None):
        # Sets logging options. Note that logging increases the execution time,
        # as the final GM has to be build each time
        # Paramters:
        #   logger: GMLogger
        #       GMLogger object to call every iteration
        #
        self._logger = logger

    def generate(self, pcbatch: torch.Tensor, gmbatch: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        # Gets a point cloud batch of size [1,n,3] (NOTE THAT BATCH SIZE NEEDS TO BE ONE!)
        # where n is the point count.
        # If the given logger uses a scaler, the point cloud has to be given downscaled.
        # gmbatch has to be None. This generator does not support improving existing GMMs
        # It returns two gaussian gmc.mixtures the first being a mixture with amplitudes as weights
        # the second a mixture where the weights describe the priors.
        # Training parameters have to be set in the other methods of the class

        assert (gmbatch is None), "EckartGenerator cannot improve existing GMMs"

        batch_size = pcbatch.shape[0]
        assert (batch_size is 1), "EckartGenerator currently does not support batchsizes > 1"
        point_count = pcbatch.shape[1]
        pcbatch = pcbatch.to(self._dtype).cuda()

        # eps is used to avoid singularities and as default covariance for invalid gaussians
        self._eps = (torch.eye(3, 3, dtype=self._dtype) * 1e-4).view(1, 1, 1, 3, 3).cuda()

        # parent_per_point (1,n) identifies which gaussian in the previous layer this point is assigned to
        # parent_per_point = torch.zeros(1, point_count).to(torch.long).cuda()
        # the 0th layer, only has one (fictional) Gaussian, whose index is assumed to be 0
        # parent_per_point[:, :] = 0
        # the p_ik. How much influence each point has to each parent. sum for each point is 1
        point_weighting_factors = torch.ones(point_count, 1, dtype=self._dtype).cuda()

        # hierarchy: list of combined mixtures, one for each level
        hierarchy = []
        # weights of the parents of each point. initialized with one (fictional 0th layer has only one Gaussian)
        parentweights = torch.ones(1, 1, self._n_gaussians_per_node, dtype=self._dtype).cuda()

        llh_loss_calc = LikelihoodLoss()

        absiteration = 0  # Iteration index overall
        for level in range(self._n_levels):
            print("Level: ", level)
            parentcount_for_level = self._n_gaussians_per_node ** level  # How many parents the current level has
            childcount_for_level = parentcount_for_level * self._n_gaussians_per_node
            relevant_parents = torch.arange(0, parentcount_for_level).cuda()  # (parentcount)

            # Scaler, to scale down the sub-pointclouds, and up the resulting sub-gms
            bbs = self.extract_bbs(pcbatch, point_weighting_factors) # (K,2,3)

            # Initialize GMs
            gm_data = self._initialize_gms_on_bounding_box(bbs, relevant_parents, parentweights)

            self._termination_criterion.reset()

            iteration = 0  # Iteration index for this level
            # EM-Loop
            while True:
                iteration += 1
                absiteration += 1

                points = pcbatch.unsqueeze(1).unsqueeze(3)  # (1, 1, np, 1, 3)

                # E-Step
                responsibilities = self._expectation(points, gm_data, point_weighting_factors, relevant_parents)
                # losses = scaler.scale_up_losses(losses)
                # Nan-Losses cannot be avoided and will be ignored for the mean calculation
                mixture = gm_data.approximate_whole_mixture()
                loss = llh_loss_calc.calculate_score_packed(pcbatch, mixture)
                # loss = -losses[~torch.isnan(losses)].mean().view(1) # ToDo, look at this
                # del losses

                pwf_per_gaussian = point_weighting_factors.unsqueeze(2) \
                    .expand(point_count, parentcount_for_level, self._n_gaussians_per_node).reshape(point_count, -1)
                weighted_responsibilities = pwf_per_gaussian * responsibilities

                if self._logger:
                    # For logging, we do not actually get the accurate mixture, but we approximate it, as this is faster
                    mixture = gm_data.approximate_whole_mixture()
                    self._logger.log(absiteration - 1, loss, mixture)
                    del mixture

                if not self._termination_criterion.may_continue(iteration - 1, loss.view(-1)).item():
                    # Parition: Update point_weighting_factors
                    new_pwf = torch.zeros_like(responsibilities)    # (1, 1, np, ng)
                    new_pwf[responsibilities > self._partition_threshold] = \
                        weighted_responsibilities[responsibilities > self._partition_threshold]
                    new_pwf /= new_pwf.sum(dim=3, keepdim=True) # normalize
                    new_pwf[torch.isnan(new_pwf)] = 0.0 # fix NaNs for points that are assigned to no Gaussian
                    point_weighting_factors = new_pwf.view(point_count, -1)
                    break

                # M-Step
                self._maximization(points, weighted_responsibilities, pwf_per_gaussian, gm_data)

            # add gm to hierarchy
            hierarchy.append(gm_data)
            # update parentweights
            parentweights = gm_data.get_premultiplied_priors().repeat(1, 1, self._n_gaussians_per_node, 1)\
                .transpose(-1, -2).reshape(1, 1, -1)

        # Calculate final GMs
        res_gm, res_gmm = self._construct_gm_from_hierarchy(hierarchy)
        res_gm = res_gm.float()
        res_gmm = res_gmm.float()

        print("Final Loss: ", LikelihoodLoss().calculate_score_packed(pcbatch, res_gm).item())
        # print("EckartSP: # of invalid Gaussians: ", torch.sum(gm.weights(res_gmm).eq(0)).item())

        return res_gm, res_gmm

    def extract_bbs(self, pcbatch: torch.Tensor, point_weighting_factors: torch.Tensor) -> torch.Tensor:
        # This gets a pcbatch of size (1, n, 3) and point_weighting_factors of size (n, K) (K = #parents)
        # This returns the bounding boxes for each parent (K,2,3) (0=min,1=extend)
        n_points = pcbatch.shape[1]
        n_parents = point_weighting_factors.shape[1]
        # pwf_per_gaussian = point_weighting_factors.repeat(1, 1, n_gaussians_per_parent).view(n_points, -1) # (n, K*j)
        pwfp_relevant = point_weighting_factors > 0
        result = torch.zeros(n_parents, 2, 3, dtype=pcbatch.dtype).cuda()
        for i in range(n_parents):
            rel_points = pcbatch[0, pwfp_relevant[:, i], :]
            if rel_points.shape[0] > 1:
                rel_bbmin = torch.min(rel_points, dim=0)[0]
                rel_bbmax = torch.max(rel_points, dim=0)[0]
                result[i, 0, :] = rel_bbmin
                result[i, 1, :] = rel_bbmax - rel_bbmin
                zeroextend = (result[i, 1, :] == 0)
                if zeroextend.any():
                    result[i, 0, zeroextend] -= 0.01
                    result[i, 1, zeroextend] = 0.02
            # elif rel_points.shape[0] == 1:
            #     result[i, 0, :] = rel_points[0] - torch.tensor([0.01, 0.01, 0.01]).cuda()
            #     result[i, 1, :] = torch.tensor([0.02, 0.02, 0.02])
            else:
                result[i, 0, :] = torch.tensor([0.0, 0.0, 0.0])
                result[i, 1, :] = torch.tensor([1.0, 1.0, 1.0])
        return result

    def _initialize_gms_on_bounding_box(self, bbs: torch.Tensor, relevant_parents: torch.Tensor, parentweights: torch.Tensor):
        # ToDo: At the moment the bounding box might not be a regular cube. this may or may not be okay.
        gmcount = bbs.shape[0]
        gmdata = self.GMLevelTrainingData(self._dtype)
        gmdata.parents = relevant_parents.repeat(1, self._n_gaussians_per_node)\
            .view(self._n_gaussians_per_node, -1).transpose(-1, -2).reshape(1, 1, -1)
        bbs_rep = bbs.unsqueeze(0).repeat(self._n_gaussians_per_node, 1, 1, 1).transpose(0, 1).reshape(-1, 2, 3)
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
            gmdata.positions = position_templates[0:self._n_gaussians_per_node].unsqueeze(0).unsqueeze(0)\
                .repeat(1, 1, gmcount, 1)
        else:
            gmdata.positions = position_templates.unsqueeze(0).unsqueeze(0). \
                repeat(1, 1, math.ceil(self._n_gaussians_per_node / 8), 1)
            gmdata.positions = gmdata.positions[:, :, 0:self._n_gaussians_per_node, :].repeat(1, 1, gmcount, 1)
        gmdata.positions[0, 0, :, :] *= bbs_rep[:, 1, :]
        gmdata.positions[0, 0, :, :] += bbs_rep[:, 0, :]
        gmdata.covariances = 0.1 * torch.eye(3).to(self._dtype).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(0). \
            repeat(1, 1, self._n_gaussians_per_node * gmcount, 1, 1)
        gmdata.covariances[0, 0, :, :, :] *= bbs_rep[:, 1, :].unsqueeze(2) ** 2
        gmdata.priors = torch.zeros(1, 1, self._n_gaussians_per_node * gmcount).to(self._dtype).cuda()
        gmdata.priors[:, :, :] = 1 / self._n_gaussians_per_node

        gmdata.parentweights = parentweights

        return gmdata

    def _expectation(self, points: torch.Tensor, gm_data, point_weighting_factors: torch.Tensor,
                     relevant_parents: torch.Tensor) -> torch.Tensor:
        # This performs the Expectation step of the EM Algorithm. This calculates 1) the responsibilities.
        # So the probabilities, how likely each point belongs to each gaussian and 2) the overall Log-Likelihoods
        # for each Sub-GM given the point cloud.
        # The calculations are performed numerically stable in Log-Space!
        # Parameters:
        #   points_rep: torch.Tensor of shape (1, 1, n_points, 1, 3)
        #       This is a view of the point cloud
        #   gm_data: GMLevelTrainingData
        #       The current GM-object
        #   point_weighting_factors: torch.Tensor of shape (n,K)
        #       The per-point-weighting-factor of each point to each parent
        #   relevant_parents: torch.Tensor of shape (p)
        #       List of all relevant parents. Should be an aranged list like [0,1,2,3,4,5,6,7]
        # Returns:
        #   responsibilities: torch.Tensor of shape (1, 1, n, p*g) where g is self._n_gaussians_per_node (results in
        #       the number of all gaussians on this level)
        #       responsibilities for point-gaussian-combinations where the point does not belong to a
        #       gaussian's parent will be 0. Also note, that there might be Sub-GMs without points assigned to them.
        #       The losses of these will be nan!
        #   This does not return losses, as the actual loss might drift far away from this loss
        #
        #   Note that this does not support executing only parts of the responsibilities at once for memory optimization
        #   (as in the M-Step). It would be possible to implement this though.

        batch_size = points.shape[0]
        n_sample_points = points.shape[2]
        all_gauss_count = gm_data.positions.shape[2]
        parent_count = relevant_parents.shape[0]

        # mask_indizes is a list of indizes of a) points with their corresponding b) gauss (child) indizes
        mask_indizes = torch.nonzero(point_weighting_factors, as_tuple=False)
        mask_indizes = mask_indizes.repeat(1, 1, self._n_gaussians_per_node)
        mask_indizes[:, :, 1::2] *= self._n_gaussians_per_node
        mask_indizes[:, :, 1::2] += torch.arange(0, self._n_gaussians_per_node, dtype=torch.long).cuda()
        mask_indizes = mask_indizes.view(-1, 2)
        pairing_count = mask_indizes.shape[0]

        # points_rep: shape (1, 1, xc, 3)
        points_rep = torch.zeros(1, 1, pairing_count, 3, dtype=self._dtype).cuda()
        points_rep[0, 0, :, :] = points[0, 0, mask_indizes[:, 0], 0, :]
        # GM-Positions, expanded for each PC point. shape: (1, 1, xc, 3)
        gmpositions_rep = torch.zeros_like(points_rep)
        gmpositions_rep[0, 0, :, :] = gm_data.positions[0, 0, mask_indizes[:, 1], :]
        # GM-Inverse Covariances, expanded for each PC point. shape: (1, 1, xc, 3, 3)
        gmicovs_rep = torch.zeros(1, 1, pairing_count, 3, 3, dtype=self._dtype).cuda()
        gmicovs_rep[0, 0, :, :, :] = gm_data.calculate_inversed_covariances()[0, 0, mask_indizes[:, 1], :, :]
        # Tensor of {PC-point minus GM-position}-vectors. shape: (1, 1, xc, 3, 1)
        grelpos = (points_rep - gmpositions_rep).unsqueeze(4)
        # Tensor of 0.5 times the Mahalanobis distances of PC points to Gaussians. shape: (1, 1, xc)
        expvalues = 0.5 * \
            torch.matmul(grelpos.transpose(-2, -1), torch.matmul(gmicovs_rep, grelpos)).squeeze(4).squeeze(3)
        # Logarithmized GM-Priors, expanded for each PC point. shape: (1, 1, xc)
        gmpriors_log_rep = torch.zeros(1, 1, pairing_count, dtype=self._dtype).cuda()
        gmpriors_log_rep[0, 0, :] = torch.log(gm_data.calculate_amplitudes()[0, 0, mask_indizes[:, 1]])
        parentpriors_log_rep = torch.zeros_like(gmpriors_log_rep)
        parentpriors_log_rep[0, 0, :] = torch.log(gm_data.parentweights[0, 0, mask_indizes[:, 1]])

        # The logarithmized likelihoods of each point for each gaussian. shape: (1, 1, xc)
        likelihood_log = gmpriors_log_rep - expvalues
        global_likelihood_log = parentpriors_log_rep + likelihood_log

        # Logarithmized Likelihood for each point given the GM. shape: (1, 1, np, 1)
        llh_intermediate = torch.zeros(n_sample_points, all_gauss_count, dtype=self._dtype).cuda()
        llh_intermediate[:, :] = -float("inf")
        llh_intermediate[mask_indizes[:, 0], mask_indizes[:, 1]] = likelihood_log.view(-1)
        llh_sum = torch.logsumexp(llh_intermediate, dim=1).view(1, 1, n_sample_points, 1)
        # llh_intermediate[mask_indizes[:, 0], mask_indizes[:, 1]] = global_likelihood_log.view(-1)
        # global_llh_sum = torch.logsumexp(llh_intermediate, dim=1).view(n_sample_points, 1) # loss per point
        del llh_intermediate

        # prefilter = relevant_parents.unsqueeze(1).expand(n_sample_points, pairing_count)
        # lhfilter = (prefilter == mask_indizes[:, 0].unsqueeze(0).expand(n_sample_points, pairing_count))
        # llh_repl = likelihood_log.expand(1, n_sample_points, pairing_count) * lhfilter
        # llh_sum = torch.logsumexp(llh_repl, dim=1).view(1, 1, n_sample_points, 1)
        # del llh_repl
        # gllh_repl = global_likelihood_log.expand(1, n_sample_points, pairing_count) * lhfilter
        # global_llh_sum = torch.logsumexp(gllh_repl, dim=1).view(1, 1, n_sample_points, 1)
        # del gllh_repl

        # llh_sum = torch.zeros(1, 1, n_sample_points, 1, dtype=self._dtype).cuda()
        # global_llh_sum = torch.zeros_like(llh_sum)
        # for p in range(n_sample_points):
        #     relevant_pairs = mask_indizes[:, 0] == p
        #     llh_sum[0, 0, p, 0] = torch.logsumexp(likelihood_log[0, 0, relevant_pairs], dim=0)
        #     global_llh_sum[0, 0, p, 0] = torch.logsumexp(global_likelihood_log[0, 0, relevant_pairs], dim=0)

        # Calculate Loss per GM (per Parent)
        # indicator = torch.greater(point_weighting_factors, 0)  # indicator: (np, nP)
        # losses = ((indicator * global_llh_sum.repeat(1, parent_count)).nansum(dim=0)
        #           / indicator.sum(dim=0)).view(1, -1)  # (1, nP)

        # Responsibilities_flat: (1, 1, xc)
        responsibilities_flat = torch.exp(likelihood_log - llh_sum[:, :, mask_indizes[:, 0], 0])
        responsibilities_matr = torch.zeros(1, 1, n_sample_points, all_gauss_count).cuda()
        responsibilities_matr[0, 0, mask_indizes[:, 0], mask_indizes[:, 1]] = responsibilities_flat

        return responsibilities_matr# , losses

    def _maximization(self, points: torch.Tensor, weighted_responsibilities: torch.Tensor, pwf_per_gaussian: torch.Tensor, gm_data):
        # This performs the Maximization step of the EM Algorithm.
        # Updates the GM-Model given the responsibilities which resulted from the E-Step.
        # Per default, all points and Gaussians are processed at once.
        # However, by setting m_step_gaussians_subbatchsize and m_step_points_subbatchsize in the constructor,
        # this can be split into several processings to save memory.
        # Parameters:
        #   points: torch.Tensor of shape (1, 1, n_points, 1, 3)
        #       View of the point cloud
        #   weighted_responsibilities: torch.Tensor of shape (1, 1, n_points, n_gaussians)
        #       This is the result of the E-step, multiplied with the pwfs
        #   pwf_per_gaussian: torch.Tensor of shape (n_points, n_gaussian)
        #       point weighting factors per parent (for each gaussian)
        #   gm_data: TrainingData
        #       The current GM-object (will be changed)

        n_sample_points = points.shape[2]
        all_gauss_count = weighted_responsibilities.shape[3]
        gauss_subbatch_size = self._n_gaussians_per_node * self._m_step_gaussians_subbatchsize
        if gauss_subbatch_size < 1:
            gauss_subbatch_size = all_gauss_count
        point_subbatch_size = self._m_step_points_subbatchsize
        if point_subbatch_size < 1:
            point_subbatch_size = n_sample_points

        # Iterate over Gauss-Subbatches
        for j_start in range(0, all_gauss_count, gauss_subbatch_size):
            j_end = j_start + gauss_subbatch_size
            actual_gauss_subbatch_size = min(all_gauss_count, j_end) - j_start
            # Initialize T-Variables for these Gaussians, will be filled in the upcoming loop
            # Positions/Covariances/Priors are calculated from these (see Eckart-Paper)
            t_0 = torch.zeros(1, 1, actual_gauss_subbatch_size, dtype=self._dtype).cuda()
            t_1 = torch.zeros(1, 1, actual_gauss_subbatch_size, 3, dtype=self._dtype).cuda()
            t_2 = torch.zeros(1, 1, actual_gauss_subbatch_size, 3, 3, dtype=self._dtype).cuda()

            # Iterate over Point-Subbatches
            for i_start in range(0, n_sample_points, point_subbatch_size):
                i_end = i_start + point_subbatch_size
                relevant_responsibilities = weighted_responsibilities[:, :, i_start:i_end, j_start:j_end]
                actual_gauss_subbatch_size = relevant_responsibilities.shape[3]
                actual_point_subbatch_size = relevant_responsibilities.shape[2]

                points_rep = points[:, :, i_start:i_end]\
                    .expand(1, 1, actual_point_subbatch_size, actual_gauss_subbatch_size, 3)
                matrices_from_points = points_rep.unsqueeze(5) * points_rep.unsqueeze(5).transpose(-1, -2)

                # Fill T-Variables
                t_2 += (matrices_from_points[:, :, :, 0:actual_gauss_subbatch_size]
                        * relevant_responsibilities.unsqueeze(4).unsqueeze(5)).sum(dim=2)  # shape: (1, 1, J, 3, 3)
                t_0 += relevant_responsibilities.sum(dim=2)  # shape: (1, 1, J)
                t_1 += (points_rep[:, :, :, 0:actual_gauss_subbatch_size] * relevant_responsibilities.unsqueeze(4))\
                    .sum(dim=2)  # shape: (1, 1, J, 3)
                del matrices_from_points

            # formulas taken from eckart paper
            gm_data.positions[:, :, j_start:j_end] = t_1 / t_0.unsqueeze(3)  # (1, 1, J, 3)
            gm_data.covariances[:, :, j_start:j_end] = t_2 / t_0.unsqueeze(3).unsqueeze(4) - \
                (gm_data.positions[:, :, j_start:j_end].unsqueeze(4) *
                    gm_data.positions[:, :, j_start:j_end].unsqueeze(4).transpose(-1, -2)) \
                + self._eps.expand_as(t_2)
            relevant_point_count = pwf_per_gaussian[:, j_start:j_end].sum(dim=0).view(1, -1) # (1, J)
            gm_data.priors[:, :, j_start:j_end] = t_0 / relevant_point_count
            del t_0, t_1, t_2

        # Handling of invalid Gaussians! If all responsibilities of a Gaussian are zero, the previous code will
        # set the prior of it to zero and the covariances and positions to NaN
        # To avoid NaNs, we will then replace those invalid values with 0 (pos) and eps (cov).
        nans = torch.isnan(gm_data.priors) | (gm_data.priors == 0)
        gm_data.positions[nans] = torch.tensor([0.0, 0.0, 0.0], dtype=self._dtype).cuda()
        gm_data.covariances[nans] = self._eps[0, 0, 0, :, :]
        gm_data.priors[nans] = 0

    def _construct_gm_from_hierarchy(self, hierarchy) -> (torch.Tensor, torch.Tensor):
        # Constructs a final GM from a level hierarchy
        # Returns the same mixture once with amplitudes as weights, once with priors
        priors, covariances, positions = self._expand_subgm_from_level(hierarchy, -1, 0)
        resgmm = gm.pack_mixture(priors, positions, covariances)
        resgm = gm.convert_priors_to_amplitudes(resgmm)
        return resgm, resgmm

    def _expand_subgm_from_level(self, hierarchy, level, index) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        # Helperfunction for _construct_gm_from_hierarchy
        # Expands a subgm from a certain node.
        if level == len(hierarchy) - 1:
            gmdata = hierarchy[-1]
            return gmdata.priors[:, :, index:index + 1], gmdata.covariances[:, :, index:index + 1], \
                gmdata.positions[:, :, index:index + 1]
        else:
            gmdata_upper = hierarchy[level]
            myweight = 1
            if level != -1:
                myweight = gmdata_upper.priors[:, :, index]
            if myweight == 0:
                return gmdata_upper.priors[:, :, index:index + 1], \
                       gmdata_upper.covariances[:, :, index:index + 1], \
                       gmdata_upper.positions[:, :, index:index + 1]
            restlevels = len(hierarchy) - level - 1
            maxgaussians = self._n_gaussians_per_node ** restlevels
            subgm_priors = torch.zeros(1, 1, maxgaussians, dtype=self._dtype).cuda()
            subgm_covariances = torch.zeros(1, 1, maxgaussians, 3, 3, dtype=self._dtype).cuda()
            subgm_positions = torch.zeros(1, 1, maxgaussians, 3, dtype=self._dtype).cuda()
            currentindex = 0
            endindex = -1
            for relchildindex in range(self._n_gaussians_per_node):
                abschildindex = index * self._n_gaussians_per_node + relchildindex
                priors, covariances, positions = self._expand_subgm_from_level(hierarchy, level + 1, abschildindex)
                num = priors.shape[2]
                endindex = currentindex + num
                subgm_priors[:, :, currentindex:endindex], subgm_covariances[:, :, currentindex:endindex], \
                    subgm_positions[:, :, currentindex:endindex] = priors, covariances, positions
                currentindex = endindex
            if subgm_priors.sum() != 0:
                subgm_priors *= myweight
                return subgm_priors[:, :, 0:endindex], \
                    subgm_covariances[:, :, 0:endindex], \
                    subgm_positions[:, :, 0:endindex]
            else:
                if level == -1:
                    empty = torch.zeros(1, 1, 0, dtype=self._dtype).cuda()
                    return empty, empty.unsqueeze(3).unsqueeze(4).repeat(1, 1, 0, 3, 3), \
                        empty.unsqueeze(3).repeat(1, 1, 0, 3)
                return gmdata_upper.priors[:, :, index:index + 1], \
                    gmdata_upper.covariances[:, :, index:index + 1], \
                    gmdata_upper.positions[:, :, index:index + 1]

    class GMLevelTrainingData:
        # Helper class. Capsules all relevant training data of the current GM batch on the given level.
        # positions, covariances and priors are stored as-is and can be set.
        # inversed covariances and amplitudes can be calculated from these.
        # Additionally, for each Gaussian, its parent Gauss index, and the product of all parents priors
        # are stored.
        # Note that this is not one GM, but a whole bunch of GMs managed in the same structure

        def __init__(self, dtype):
            self.positions: torch.Tensor = torch.tensor([], dtype=dtype)
            self.priors: torch.Tensor = torch.tensor([], dtype=dtype)
            self.covariances: torch.Tensor = torch.tensor([], dtype=dtype)
            self.parents = torch.tensor([], dtype=dtype)  # (1, 1, g) Indizes of parent Gaussians on parent Level
            self.parentweights = torch.tensor([], dtype=dtype)  # (1, 1, g) Combined prior-weights of all parents

        def calculate_inversed_covariances(self) -> torch.Tensor:
            return self.covariances.inverse().contiguous()

        def get_premultiplied_priors(self) -> torch.Tensor:
            # Returns the priors multiplied with all their parents priors
            return self.parentweights * self.priors

        def calculate_amplitudes(self) -> torch.Tensor:
            return self.priors / (self.covariances.det().sqrt() * 15.74960995)

        def approximate_whole_mixture(self) -> torch.Tensor:
            # Scales the mixtures up and multiplies the priors with their parents priors
            # to generate a (more or less) valid Gaussian Mixture (with amplitudes).
            # It's not completely accurate, as all children of a Gaussian can have priors 0.
            # Usually these would be replaced with their parent. This is not happening.
            # It could therefore even be, that the weights do not sum to 0, so it's only an approximation.
            # newpriors, newpositions, newcovariances = \
            #     scaler.scale_up_gmm_wpc(self.priors, self.positions, self.covariances)
            # newamplitudes = newpriors / (newcovariances.det().sqrt() * 15.74960995)
            return gm.pack_mixture(self.parentweights * self.calculate_amplitudes(), self.positions, self.covariances)

        def scale_up(self, scaler: LevelScaler):
            # Scales the Gaussian up given the LevelScaler
            self.priors, self.positions, self.covariances = \
                scaler.scale_up_gmm_wpc(self.priors, self.positions, self.covariances)

        def __len__(self):
            # Returs the number of Gaussians in this level
            return self.priors.shape[2]
