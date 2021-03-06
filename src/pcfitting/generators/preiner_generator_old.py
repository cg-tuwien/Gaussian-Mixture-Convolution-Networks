import numpy

from gmc import mat_tools
from pcfitting import GMMGenerator, GMLogger
from pcfitting import TerminationCriterion, MaxIterationTerminationCriterion
from pcfitting.error_functions import LikelihoodLoss
from .em_tools import EMTools
from .level_scaler import LevelScaler
import torch
import gmc.mixture as gm
from gmc import mat_tools
import math


class PreinerGenerator(GMMGenerator):
    # Previous unfinished implementation of geometrically regularized Bottom-Up-HEM
    # use preiner_generator.py instead

    def __init__(self,
                 reduction_factor: float,
                 n_levels: int,
                 gauss_subbatchsize: int = -1,
                 max_gauss_pairings: int = 250000,
                 termination_criterion: TerminationCriterion = MaxIterationTerminationCriterion(20),
                 dtype: torch.dtype = torch.float32,
                 eps: float = 1e-7,
                 eps_is_relative: bool = True):
        # Constructor. Creates a new EckartGenerator.
        # Parameters:
        #   n_gaussians_per_node: int
        #       To how many Gaussians each Gaussian should be expanded to in the next level. Values higher than 8
        #       won't make much sense, as Gaussians will be initialized at the same position.
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
        #   eps: float
        #       Small value to be added to the Covariances for numerical stability
        self._reduction_factor = reduction_factor
        self._n_levels = n_levels
        self._gaussian_subbatchsize = gauss_subbatchsize
        self._max_gauss_pairings = max_gauss_pairings
        self._termination_criterion = termination_criterion
        self._dtype = dtype
        self._logger = None
        self._epsvar = eps
        if eps < 1e-9:
            print("Warning! Very small eps! Might cause numerical issues!")
        self._eps_is_relative = eps_is_relative

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

        assert (gmbatch is None), "PreinerGenerator cannot improve existing GMMs" # ToDo: maybe it can

        batch_size = pcbatch.shape[0]
        assert (batch_size is 1), "PreinerGenerator currently does not support batchsizes > 1"
        point_count = pcbatch.shape[1]
        pcbatch = pcbatch.to(self._dtype).cuda()

        epsilons = torch.ones(batch_size, dtype=self._dtype, device='cuda') * self._epsvar
        if self._eps_is_relative:
            extends = pcbatch.max(dim=1)[0] - pcbatch.min(dim=1)[0]
            epsilons *= extends.max(dim=1)[0] ** 2
            epsilons[epsilons < 1e-9] = 1e-9

        # eps is a small multiple of the identity matrix which is added to the cov-matrizes
        # in order to avoid singularities
        eps = (torch.eye(3, 3, dtype=self._dtype, device='cuda')).view(1, 1, 1, 3, 3) \
                  .expand(batch_size, 1, 1, 3, 3) * epsilons.view(-1, 1, 1, 1, 1)

        # parent_per_point (1,n) identifies which gaussian in the previous layer this point is assigned to
        parent_per_point = torch.zeros(1, point_count, dtype=torch.long, device='cuda')
        # the 0th layer, only has one (fictional) Gaussian, whose index is assumed to be 0
        parent_per_point[:, :] = 0

        llh_loss_calc = LikelihoodLoss(False)

        # hierarchy: list of combined mixtures, one for each level
        gm_ref = self._initialize_gm_on_lowest_level(pcbatch)
        gm_ref_searchradii = gm_ref.calculate_search_radii(2.5) # ToDo: Alpha

        if self._logger:
            mixture = gm_ref.pack_mixture()
            loss = llh_loss_calc.calculate_score_packed(pcbatch, mixture)
            self._logger.log(0, loss, mixture)

        absiteration = 1  # Iteration index overall
        for level in range(self._n_levels):
            print("Level: ", level)
            gausscount_for_level = round(point_count / (self._reduction_factor ** (level+1)))  # How many Gaussians the current level has

            # Initialize GMs
            gm_fit = self._initialize_level(gm_ref, gausscount_for_level)

            self._termination_criterion.reset()

            iteration = 0  # Iteration index for this level
            # EM-Loop
            while True:
                iteration += 1
                absiteration += 1

                # points = points.unsqueeze(1).unsqueeze(3)  # (1, 1, np, 1, 3)

                # We only approximate the whole mixture, actual loss might be better!
                mixture = gm_fit.pack_mixture()
                loss = llh_loss_calc.calculate_score_packed(pcbatch, mixture)

                if self._logger:
                    self._logger.log(absiteration - 1, loss, mixture)
                    del mixture

                if not self._termination_criterion.may_continue(iteration - 1, loss.view(-1)).item():
                    # update parent_per_point for next level from responsibilities
                    break

                # E-Step
                responsibilities = self._expectation(gm_fit, gm_ref, point_count)

                # M-Step
                self._maximization(gm_fit, gm_ref, responsibilities, point_count, eps)

            # add gm to hierarchy
            gm_ref = gm_fit

        # Calculate final GMs
        res_gm = gm_fit.pack_mixture()
        res_gmm = gm_fit.pack_mixture_model()
        res_gm = res_gm.float()
        res_gmm = res_gmm.float()

        # print("Final Loss: ", LikelihoodLoss().calculate_score_packed(pcbatch, res_gm).item())

        return res_gm, res_gmm

    def _initialize_gm_on_lowest_level(self, pcbatch: torch.Tensor):
        # For now: just put a small gaussian on each point
        # Should be adapted according to Preiner
        gmdata = self.GMLevelTrainingData(pcbatch.shape[1], pcbatch.dtype)
        gmdata.positions[:, 0, :, :] = pcbatch
        gmdata.covariances[:, 0, :, :, :] = torch.eye(3, dtype=self._dtype, device='cuda').view(1, 1, 3, 3)# self._eps.expand_as(gmdata.covariances)
        gmdata.inverse_covariances = mat_tools.inverse(gmdata.covariances)
        gmdata.priors[:, 0, :] = 1.0 / pcbatch.shape[1]
        return gmdata

    def _initialize_level(self, gm_ref, n_gaussians: int):

        sample_point_idz = torch.randperm(len(gm_ref))[0:n_gaussians]  # Shape: (s)
        gm_new = self.GMLevelTrainingData(len(gm_ref), self._dtype)
        gm_new.positions = gm_ref.positions[:, :, sample_point_idz, :].clone()
        gm_new.covariances = gm_ref.covariances[:, :, sample_point_idz, :, :].clone()
        gm_new.inverse_covariances = gm_ref.inverse_covariances[:, :, sample_point_idz, :, :].clone()
        gm_new.priors = gm_ref.priors[:, :, sample_point_idz].clone()
        gm_new.priors /= gm_new.priors.sum(2)
        return gm_new

        # point_count = points.shape[1]
        #
        # # Calculate the mean pc position. shape: (bs, 1, 3)
        # meanpos = points.mean(dim=1, keepdim=True)
        # # Calcualte (point - meanpoint) pairs. Shape: (bs, np, 3, 1)
        # diffs = (points - meanpos.expand(1, point_count, 3)).unsqueeze(3)
        # # Squeeze meanpos -> shape: (bs, 3)
        # meanpos = meanpos.squeeze(1)
        # # Calculate expected covariance. Shape: (bs, 3, 3)
        # meancov = (diffs * diffs.transpose(-1, -2)).mean(dim=[1])
        # # Calculated mean prior.
        # meanweight = 1.0 / n_gaussians
        #
        # # Sample positions from Gaussian -> shape: (bs, 1, ng, 3)
        # positions = torch.zeros(1, 1, n_gaussians, 3).to(self._dtype)
        # positions[0, 0, :, :] = torch.tensor(
        #     numpy.random.multivariate_normal(meanpos[0, :].cpu(), meancov[0, :, :].cpu(), n_gaussians), device='cuda')
        # # Repeat covariances for each Gaussian -> shape: (bs, 1, ng, 3, 3)
        # covariances = meancov.view(1, 1, 1, 3, 3).expand(1, 1, n_gaussians, 3, 3)
        # # Set weight for each Gaussian -> shape: (bs, 1, ng)
        # weights = torch.zeros(1, 1, n_gaussians).to(self._dtype)
        # weights[:, :, :] = meanweight
        #
        # gm_data = self.GMLevelTrainingData(n_gaussians, self._dtype)
        # gm_data.positions[:] = positions
        # gm_data.covariances[:] = covariances
        # gm_data.inverse_covariances = mat_tools.inverse(gm_data.covariances)
        # gm_data.priors[:] = weights
        # return gm_data

    # def _em_step(self, gm_fit, gm_ref, n_sample_count):
    #     fit_subbatch_size = min(math.floor(self._max_gauss_pairings / len(gm_ref)), len(gm_fit))
    #
    #     n_virtual_points = n_sample_count * gm_ref.priors
    #     nonzeronvp = ~n_virtual_points.eq(0)
    #     gmrpos = gm_ref.positions.unsqueeze(3)
    #     gmrcovs = gm_ref.covariances.unsqueeze(3)
    #
    #     # Current approach: Process all of gmref at once, but only for a few gmfit-entries
    #
    #     for f_start in range(0, len(gm_fit), fit_subbatch_size):
    #         f_end = min(f_start + fit_subbatch_size, len(gm_fit))
    #         this_fit_subbatch_size = f_end - f_start
    #         gmfpos = gm_fit.positions[:, :, f_start:f_end, :]
    #         gmficov = gm_fit.inverse_covariances[:, :, f_start:f_end, :, :]
    #         gmrpos_rep = gmrpos.expand(1, 1, len(gm_ref), this_fit_subbatch_size, 3)
    #         gmfpos_rep = gmfpos.unsqueeze(2).expand(1, 1, len(gm_ref), this_fit_subbatch_size, 3)
    #         gmficov_rep = gmficov.unsqueeze(2).expand(1, 1, len(gm_ref), this_fit_subbatch_size, 3, 3)
    #         gmrcovs_rep = gmrcovs.expand(1, 1, len(gm_ref), this_fit_subbatch_size, 3, 3)
    #         grelpos = (gmrpos_rep - gmfpos_rep).unsqueeze(5)
    #         expvalues = torch.matmul(grelpos.transpose(-2, -1), torch.matmul(gmficov_rep, grelpos))\
    #             .squeeze(5).squeeze(4)
    #         expvalues += mat_tools.batched_trace(torch.matmul(gmficov_rep, gmrcovs_rep))
    #         expvalues += torch.log(torch.det(gm_fit.covariances[:, :, f_start:f_end]))
    #         expvalues += 5.513631199
    #         expvalues[nonzeronvp] /= (-2*n_virtual_points[nonzeronvp]).unsqueeze(1)
    #         expvalues[~nonzeronvp] = -float("Inf")
    #         gmfpriors_log_rep = torch.log(gm_fit.priors[:, :, f_start:f_end]).unsqueeze(2) \
    #             .expand(1, 1, len(gm_ref), this_fit_subbatch_size)
    #         expvalues += gmfpriors_log_rep # This corresponds to likelihood_log
    #         del gmfpos, gmficov, gmrpos_rep, gmficov_rep, gmrcovs_rep, grelpos, gmfpriors_log_rep
    #         llh_sum = torch.logsumexp(expvalues, dim=3)

    def _expectation(self, gm_fit, gm_ref, n_sample_count) -> torch.Tensor:
        # This performs the Expectation step of the EM Algorithm. This calculates the responsibilities.
        # So the probabilities, how likely each point belongs to each gaussian.
        # The calculations are performed numerically stable in Log-Space!
        # Parameters:
        #   gm_fit: GM of this level (l)
        #   gm_ref: GM of last level (l+1)
        # Returns:
        #   responsibilities: torch.Tensor of shape (1, 1, n, p*g) where g is self._n_gaussians_per_node (results in
        #       the number of all gaussians on this level)
        #       responsibilities for point-gaussian-combinations where the point does not belong to a
        #       gaussian's parent will be 0. Also note, that there might be Sub-GMs without points assigned to them.



        gauss_subbatch_size_fit = self._gaussian_subbatchsize
        if gauss_subbatch_size_fit < 1:
            gauss_subbatch_size_fit = len(gm_fit)
        gauss_subbatch_size_ref = self._gaussian_subbatchsize
        if gauss_subbatch_size_ref < 1:
            gauss_subbatch_size_ref = len(gm_ref)

        n_virtual_points = n_sample_count * gm_ref.priors

        likelihood_log = \
            torch.zeros(1, 1, len(gm_ref), len(gm_fit), dtype=self._dtype, device='cuda')
        gmfpos = gm_fit.positions
        gmficov = gm_fit.inverse_covariances
        gmrpos = gm_ref.positions
        gmfloga = torch.log(gm_fit.calculate_amplitudes())

        for j_start in range(0, len(gm_fit), gauss_subbatch_size_fit):
            j_end = j_start + gauss_subbatch_size_fit
            actual_gauss_subbatch_size = min(len(gm_fit), j_end) - j_start
            for i_start in range(0, len(gm_ref), gauss_subbatch_size_ref):
                i_end = i_start + gauss_subbatch_size_ref
                actual_point_subbatch_size = min(len(gm_ref), i_end) - i_start
                points_rep = gmrpos[:, :, i_start:i_end, :].unsqueeze(3) \
                    .expand(1, 1, actual_point_subbatch_size, actual_gauss_subbatch_size, 3)
                # GM-Positions, expanded for each PC point. shape: (bs, 1, np, ng, 3)
                gmpositions_rep = gmfpos[:, :, j_start:j_end].unsqueeze(2) \
                    .expand(1, 1, actual_point_subbatch_size, actual_gauss_subbatch_size, 3)
                # GM-Inverse Covariances, expanded for each PC point. shape: (bs, 1, np, ng, 3, 3)
                gmicovs_rep = gmficov[:, :, j_start:j_end].unsqueeze(2) \
                    .expand(1, 1, actual_point_subbatch_size, actual_gauss_subbatch_size, 3, 3)
                pcovs_rep = gm_ref.covariances[:, :, i_start:i_end].unsqueeze(3) \
                    .expand(1, 1, actual_point_subbatch_size, actual_gauss_subbatch_size, 3, 3)
                # Tensor of {PC-point minus GM-position}-vectors. shape: (bs, 1, np, ng, 3, 1)
                grelpos = (points_rep - gmpositions_rep).unsqueeze(5)
                # Tensor of 0.5 times the Mahalanobis distances of PC points to Gaussians. shape: (bs, 1, np, ng)
                expvalues = torch.matmul(grelpos.transpose(-2, -1), torch.matmul(gmicovs_rep, grelpos)).squeeze(
                                5).squeeze(4)

                # expvalues *= 0.5
                # gma_log_rep = gmfloga[:, :, j_start:j_end].unsqueeze(2) \
                #     .expand(1, 1, actual_point_subbatch_size, actual_gauss_subbatch_size)
                # likelihood_log[:, :, i_start:i_end, j_start:j_end] = gma_log_rep - expvalues

                expvalues += mat_tools.trace(torch.matmul(gmicovs_rep, pcovs_rep))
                expvalues += torch.log(torch.det(gm_fit.covariances[:, :, j_start:j_end]))
                expvalues += 5.513631199
                # it could be, that there are gaussians with prior-weight 0, so we need to avoid division by zero
                # however, as we are actually describing ln(x^w), we know that that is ln(0), so -inf
                relnvp = n_virtual_points[:, :, i_start:i_end] # (bs, 1, np)
                nonzeronvp = ~relnvp.eq(0)
                expvalues[nonzeronvp] /= (-2*relnvp[nonzeronvp]).unsqueeze(1)
                expvalues[~nonzeronvp] = -float("Inf")
                gmpriors_log_rep = torch.log(gm_fit.priors[:, :, j_start:j_end]).unsqueeze(2) \
                    .expand(1, 1, actual_point_subbatch_size, actual_gauss_subbatch_size)
                # ToDo: Add KL-Divergence
                # The logarithmized likelihoods of each point for each gaussian. shape: (bs, 1, np, ng)
                likelihood_log[:, :, i_start:i_end, j_start:j_end] = expvalues + gmpriors_log_rep

        llh_sum = torch.logsumexp(likelihood_log, dim=3, keepdim=True)
        responsibilities = torch.exp(likelihood_log - llh_sum) # these are the w_is

        # n_virtual_points = n_sample_count * gm_ref.priors
        # gaussian_values = gm.evaluate_componentwise(gm_fit.pack_weightless_mixture(), gm_ref.positions)
        # exp_values = torch.exp(-0.5 * mat_tools.batched_trace(mat_tools.inverse(gm_fit.covariances).unsqueeze(2) @
        #                                                       gm_ref.covariances.unsqueeze(3)))
        # likelihoods = torch.pow(gaussian_values * exp_values, n_virtual_points.unsqueeze(-1))
        # likelihoods = likelihoods * gm_fit.priors.unsqueeze(2)
        # # ToDo: Add KL-Divergence
        # likelihood_sum = likelihoods.sum(3, keepdim=True)
        # responsibilities = likelihoods / (likelihood_sum + 0.0001)

        print ("#0-Res", (responsibilities.eq(0)).sum().item(), " / ", (responsibilities.shape[2] * responsibilities.shape[3]))
        assert (not torch.isnan(responsibilities).any())
        # assert not torch.sum(responsibilities, 2).eq(0).any()

        return responsibilities

    def _maximization(self, gm_fit, gm_ref, responsibilities: torch.Tensor, n_sample_count, eps):
        # This performs the Maximization step of the EM Algorithm.
        # Updates the GM-Model given the responsibilities which resulted from the E-Step.
        # Per default, all points and Gaussians are processed at once.
        # However, by setting m_step_gaussians_subbatchsize and m_step_points_subbatchsize in the constructor,
        # this can be split into several processings to save memory.
        # Parameters:
        #   points: torch.Tensor of shape (1, 1, n_points, 1, 3)
        #       View of the point cloud
        #   responsibilities: torch.Tensor of shape (1, 1, n_points, n_gaussians)
        #       This is the result of the E-step.
        #   gm_data: TrainingData
        #       The current GM-object (will be changed)

        # t_0 = torch.zeros(1, 1, len(gm_fit), dtype=self._dtype, device='cuda')
        # t_1 = torch.zeros(1, 1, len(gm_fit), 3, dtype=self._dtype, device='cuda')
        # t_2 = torch.zeros(1, 1, len(gm_fit), 3, 3, dtype=self._dtype, device='cuda')
        # relevant_points = gm_ref.positions.unsqueeze(3).expand(1, 1, len(gm_ref), len(gm_fit), 3)
        # matrices_from_points = relevant_points.unsqueeze(5) * relevant_points.unsqueeze(5).transpose(-1, -2)
        # # Fill T-Variables      # t_2 shape: (1, 1, J, 3, 3)
        # t_2 += (matrices_from_points * responsibilities.unsqueeze(4).unsqueeze(5)).sum(dim=2)
        # t_0 += responsibilities.sum(dim=2)  # shape: (1, 1, J)
        # t_1 += (relevant_points * responsibilities.unsqueeze(4)) \
        #     .sum(dim=2)  # shape: (1, 1, J, 3)
        # new_positions = t_1 / t_0.unsqueeze(3)  # (bs, 1, ng, 3)
        # new_covariances = t_2 / t_0.unsqueeze(3).unsqueeze(4) - \
        #     (new_positions.unsqueeze(4) * new_positions.unsqueeze(4).transpose(-1, -2)) + self._eps.unsqueeze(0)
        # new_priors = t_0 / len(gm_ref)

        n_virtual_points = n_sample_count * gm_ref.priors
        new_priors = torch.sum(responsibilities, 2) / len(gm_ref)
        #1, 1, nr, nf
        # This is the M-Step according to Vasconcelos, not Preiner, just for testing if this works
        weights = responsibilities * n_virtual_points.view(1, 1, len(gm_ref), 1).expand(1, 1, len(gm_ref), len(gm_fit))
        new_positions = (weights.unsqueeze(-1) * gm_ref.positions.unsqueeze(3).expand(1, 1, len(gm_ref), len(gm_fit), 3)).sum(2) / torch.sum(weights, 2).unsqueeze(-1)
        pos_diffs = gm_ref.positions.unsqueeze(3).unsqueeze(5) - new_positions.unsqueeze(2).unsqueeze(5)
        new_covariances = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * (gm_ref.covariances.unsqueeze(3) + pos_diffs.matmul(pos_diffs.transpose(-1, -2))), 2)
        new_covariances /= torch.sum(weights, 2).unsqueeze(-1).unsqueeze(-1)
        new_covariances = new_covariances + eps
        assert (new_covariances < 1e50).all()

        # weights = responsibilities * gm_ref.priors.unsqueeze(-1)
        # new_priors = torch.sum(weights, 2)
        # weights /= (new_priors + 0.0001).unsqueeze(2) # use an eps value for this
        # new_positions = torch.sum(weights.unsqueeze(-1) * gm_ref.positions.unsqueeze(3), 2)
        # pos_diffs = gm_ref.positions.unsqueeze(3).unsqueeze(5) - new_positions.unsqueeze(2).unsqueeze(5)
        # new_covariances = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * (gm_ref.covariances.unsqueeze(3) + pos_diffs.matmul(pos_diffs.transpose(-1, -2))), 2)
        # new_covariances = new_covariances + (new_priors.lt(0.0001)).unsqueeze(-1).unsqueeze(-1) * torch.eye(3, device='cuda').view(1, 1, 1, 3, 3) * 0.0001 # ToDo: eps
        gm_fit.priors = new_priors
        gm_fit.positions = new_positions
        gm_fit.set_covariances_where_valid(0, responsibilities.shape[3], new_covariances)

        # Handling of invalid Gaussians! If all responsibilities of a Gaussian are zero, the previous code will
        # set the prior of it to zero and the covariances and positions to NaN
        # To avoid NaNs, we will then replace those invalid values with 0 (pos) and eps (cov).
        nans = torch.isnan(gm_fit.priors) | (gm_fit.priors == 0)
        gm_fit.positions[nans] = torch.tensor([0.0, 0.0, 0.0], dtype=self._dtype, device='cuda')
        gm_fit.covariances[nans] = eps[0, 0, 0]
        gm_fit.priors[nans] = 0
        print("#0-Priors", gm_fit.priors.eq(0).sum().item(), "/", len(gm_fit))
        print("Sum of Priors", gm_fit.priors.sum())
        print("--")


    class GMLevelTrainingData:
        # Helper class. Capsules all relevant training data of the current GM batch on the given level.
        # positions, covariances and priors are stored as-is and can be set.
        # inversed covariances and amplitudes can be calculated from these.
        # Additionally, for each Gaussian, its parent Gauss index, and the product of all parents priors
        # are stored.
        # Note that this is not one GM, but a whole bunch of GMs managed in the same structure

        def __init__(self, count, dtype):
            self.positions: torch.Tensor = torch.zeros(1, 1, count, 3, dtype=dtype, device='cuda')
            self.priors: torch.Tensor = torch.zeros(1, 1, count, dtype=dtype, device='cuda')
            self.covariances: torch.Tensor = torch.zeros(1, 1, count, 3, 3, dtype=dtype, device='cuda')
            self.inverse_covariances: torch.Tensor = torch.zeros(1, 1, count, 3, 3, dtype=dtype, device='cuda')

        def set_covariances_where_valid(self, j_start: int, j_end: int, covariances: torch.Tensor):
            # Checks if given covariances are valid, only then they are taken over
            invcovs = mat_tools.inverse(covariances).contiguous()
            relcovs = EMTools.find_valid_matrices(covariances, invcovs)
            if (~relcovs).sum() != 0:
                print("ditching ", (~relcovs).sum().item(), " items")
            jcovariances = self.covariances[:, :, j_start:j_end]
            jcovariances[relcovs] = covariances[relcovs]
            self.covariances[:, :, j_start:j_end] = jcovariances
            jinvcovariances = self.inverse_covariances[:, :, j_start:j_end]
            jinvcovariances[relcovs] = invcovs[relcovs]
            self.inverse_covariances[:, :, j_start:j_end] = jinvcovariances

        def calculate_amplitudes(self) -> torch.Tensor:
            return self.priors * self.normal_amplitudes()

        def normal_amplitudes(self) -> torch.Tensor:
            return 1 / (self.covariances.det().sqrt() * 15.74960995)

        def pack_weightless_mixture(self) -> torch.Tensor:
            return gm.pack_mixture(self.normal_amplitudes(), self.positions, self.covariances)

        def pack_mixture(self):
            return gm.pack_mixture(self.calculate_amplitudes(), self.positions, self.covariances)

        def pack_mixture_model(self):
            return gm.pack_mixture(self.priors, self.positions, self.covariances)

        def calculate_search_radii(self, alpha) -> torch.Tensor:
            eigenvalues, _ = torch.symeig(self.covariances)
            return eigenvalues[:, -1] * alpha

        def __len__(self):
            # Returs the number of Gaussians in this level
            return self.priors.shape[2]
