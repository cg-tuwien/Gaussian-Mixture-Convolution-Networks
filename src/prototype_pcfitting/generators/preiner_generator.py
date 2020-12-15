import numpy

from gmc import mat_tools
from prototype_pcfitting import GMMGenerator, GMLogger
from prototype_pcfitting import TerminationCriterion, MaxIterationTerminationCriterion
from prototype_pcfitting.error_functions import LikelihoodLoss
from .level_scaler import LevelScaler
import torch
import gmc.mixture as gm
import math


class PreinerGenerator(GMMGenerator):

    def __init__(self,
                 reduction_factor: float,
                 n_levels: int,
                 termination_criterion: TerminationCriterion = MaxIterationTerminationCriterion(20),
                 dtype: torch.dtype = torch.float32,
                 eps: float = 1e-4):
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
        self._termination_criterion = termination_criterion
        self._dtype = dtype
        self._logger = None
        self._eps = (torch.eye(3, 3, dtype=self._dtype) * eps).view(1, 1, 3, 3).cuda()

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

        # parent_per_point (1,n) identifies which gaussian in the previous layer this point is assigned to
        parent_per_point = torch.zeros(1, point_count).to(torch.long).cuda()
        # the 0th layer, only has one (fictional) Gaussian, whose index is assumed to be 0
        parent_per_point[:, :] = 0

        llh_loss_calc = LikelihoodLoss()

        # hierarchy: list of combined mixtures, one for each level
        gm_ref = self._initialize_gm_on_lowest_level(pcbatch)

        absiteration = 0  # Iteration index overall
        for level in range(self._n_levels):
            print("Level: ", level)
            gausscount_for_level = round(point_count / (self._reduction_factor ** level))  # How many Gaussians the current level has

            # Initialize GMs
            gm_fit = self._initialize_level(pcbatch, gausscount_for_level)

            self._termination_criterion.reset()

            iteration = 0  # Iteration index for this level
            # EM-Loop
            while True:
                iteration += 1
                absiteration += 1

                # points = points.unsqueeze(1).unsqueeze(3)  # (1, 1, np, 1, 3)

                # E-Step
                responsibilities = self._expectation(gm_fit, gm_ref, point_count)
                # We only approximate the whole mixture, actual loss might be better!
                mixture = gm_fit.pack_mixture()
                loss = llh_loss_calc.calculate_score_packed(pcbatch, mixture)

                if self._logger:
                    self._logger.log(absiteration - 1, loss, mixture)
                    del mixture

                if not self._termination_criterion.may_continue(iteration - 1, loss.view(-1)).item():
                    # update parent_per_point for next level from responsibilities
                    break

                # M-Step
                self._maximization(gm_fit, gm_ref, responsibilities)

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
        gmdata.covariances[:, 0, :, :, :] = self._eps.expand_as(gmdata.covariances)
        gmdata.priors[:, 0, :] = 1.0 / pcbatch.shape[1]
        return gmdata

    def _initialize_level(self, points: torch.Tensor, n_gaussians: int):
        point_count = points.shape[1]

        # Calculate the mean pc position. shape: (bs, 1, 3)
        meanpos = points.mean(dim=1, keepdim=True)
        # Calcualte (point - meanpoint) pairs. Shape: (bs, np, 3, 1)
        diffs = (points - meanpos.expand(1, point_count, 3)).unsqueeze(3)
        # Squeeze meanpos -> shape: (bs, 3)
        meanpos = meanpos.squeeze(1)
        # Calculate expected covariance. Shape: (bs, 3, 3)
        meancov = (diffs * diffs.transpose(-1, -2)).mean(dim=[1])
        # Calculated mean prior.
        meanweight = 1.0 / n_gaussians

        # Sample positions from Gaussian -> shape: (bs, 1, ng, 3)
        positions = torch.zeros(1, 1, n_gaussians, 3).to(self._dtype)
        positions[0, 0, :, :] = torch.tensor(
            numpy.random.multivariate_normal(meanpos[0, :].cpu(), meancov[0, :, :].cpu(), n_gaussians)).cuda()
        # Repeat covariances for each Gaussian -> shape: (bs, 1, ng, 3, 3)
        covariances = meancov.view(1, 1, 1, 3, 3).expand(1, 1, n_gaussians, 3, 3)
        # Set weight for each Gaussian -> shape: (bs, 1, ng)
        weights = torch.zeros(1, 1, n_gaussians).to(self._dtype)
        weights[:, :, :] = meanweight

        gm_data = self.GMLevelTrainingData(n_gaussians, self._dtype)
        gm_data.positions[:] = positions
        gm_data.covariances[:] = covariances
        gm_data.priors[:] = weights
        return gm_data

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

        n_virtual_points = n_sample_count * gm_ref.priors
        gaussian_values = gm.evaluate_componentwise(gm_fit.pack_weightless_mixture(), gm_ref.positions)
        exp_values = torch.exp(-0.5 * mat_tools.batched_trace(mat_tools.inverse(gm_fit.covariances).unsqueeze(2) @
                                                              gm_ref.covariances.unqsueeze(3)))
        likelihoods = torch.pow(gaussian_values * exp_values, n_virtual_points.unsqueeze(-1))
        likelihoods = likelihoods * gm_fit.priors.unsqueeze(2)
        # ToDo: Add KL-Divergence
        likelihood_sum = likelihoods.sum(3, keepdim=True)
        responsibilities = likelihoods / (likelihood_sum + 0.0001)

        return responsibilities

    def _maximization(self, gm_fit, gm_ref, responsibilities: torch.Tensor):
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

        weights = responsibilities * gm_ref.priors.unsqueeze(-1)
        new_priors = torch.sum(weights, 2)
        weights /= (new_priors + 0.0001).unsqueeze(2) # use an eps value for this
        new_positions = torch.sum(weights.unsqueeze(-1) * gm_ref.positions.unsqueeze(3), 2)
        pos_diffs = gm_ref.positions.unsqueeze(3).unsqueeze(5) - new_positions.unsqueeze(2).unsqueeze(5)
        new_covariances = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * (gm_ref.covariances.unsqueeze(3) + pos_diffs.matmul(pos_diffs.transpose(-1, -2))), 2)
        new_covariances = new_covariances + (new_priors.lt(0.0001)).unsqueeze(-1).unsqueeze(-1) * torch.eye(3).cuda().view(1, 1, 1, 3, 3) * 0.0001 # ToDo: eps
        gm_fit.priors = new_priors
        gm_fit.positions = new_positions
        gm_fit.covariances = new_covariances

        # Handling of invalid Gaussians! If all responsibilities of a Gaussian are zero, the previous code will
        # set the prior of it to zero and the covariances and positions to NaN
        # To avoid NaNs, we will then replace those invalid values with 0 (pos) and eps (cov).
        nans = torch.isnan(gm_fit.priors) | (gm_fit.priors == 0)
        gm_fit.positions[nans] = torch.tensor([0.0, 0.0, 0.0], dtype=self._dtype).cuda()
        gm_fit.covariances[nans] = self._eps[0, 0, :, :]
        gm_fit.priors[nans] = 0


    class GMLevelTrainingData:
        # Helper class. Capsules all relevant training data of the current GM batch on the given level.
        # positions, covariances and priors are stored as-is and can be set.
        # inversed covariances and amplitudes can be calculated from these.
        # Additionally, for each Gaussian, its parent Gauss index, and the product of all parents priors
        # are stored.
        # Note that this is not one GM, but a whole bunch of GMs managed in the same structure

        def __init__(self, count, dtype):
            self.positions: torch.Tensor = torch.zeros(1, 1, count, 3, dtype=dtype).cuda()
            self.priors: torch.Tensor = torch.zeros(1, 1, count, dtype=dtype).cuda()
            self.covariances: torch.Tensor = torch.zeros(1, 1, count, 3, 3, dtype=dtype).cuda()

        def calculate_inversed_covariances(self) -> torch.Tensor:
            return self.covariances.inverse().contiguous()

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

        def __len__(self):
            # Returs the number of Gaussians in this level
            return self.priors.shape[2]
