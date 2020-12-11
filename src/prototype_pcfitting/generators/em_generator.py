from prototype_pcfitting import GMMGenerator, GMLogger, data_loading
from prototype_pcfitting import TerminationCriterion, MaxIterationTerminationCriterion
from gmc.cpp.extensions.furthest_point_sampling import furthest_point_sampling
import torch
import gmc.mixture as gm
import numpy


class EMGenerator(GMMGenerator):
    # GMM Generator using simple Expectation Maximization (numerically stable)

    def __init__(self,
                 n_gaussians: int,
                 n_sample_points: int = -1,
                 termination_criterion: TerminationCriterion = MaxIterationTerminationCriterion(100),
                 initialization_method: int = 0,
                 em_step_gaussians_subbatchsize: int = -1,
                 em_step_points_subbatchsize: int = -1,
                 dtype: torch.dtype = torch.float32,
                 eps: float = 1e-4):
        # Constructor. Creates a new EMGenerator.
        # Parameters:
        #   n_gaussians: int
        #       Number of components this Generator should create.
        #       This should always be set correctly, also when this is used for refining.
        #   n_sample_points: int
        #       Number of points to use each iteration. -1 if all should be used
        #   termination_criteration: TerminationCriterion
        #       Defining when to terminate. Default: After 100 Iterations.
        #       As this algorithm works on batches, the common batch loss is given to the termination criterion
        #       (We could implement saving of the best result in order to avoid moving out of optima)
        #   initialization_method: int
        #       Defines which initialization to use: 0 = Random by sample mean and cov, 1 = Random responsibilities,
        #       2 = furthest point sampling, 3 = furthest point sampling and artifical responsibilities
        #   em_step_gaussian_subbatchsize: int
        #       How many Gaussian Sub-Mixtures should be processed in the E- and M-Step at once
        #       -1 means all Gaussians (default)
        #   em_step_points_subbatchsize: int
        #       How many points should be processed in the E- and M-Step at once
        #       -1 means all Points (default)
        #   dtype: torch.dtype
        #       In which data type (precision) the operations should be performed. The final gmm is always
        #       converted to float32 though. Default: torch.float32
        #   eps: float
        #       Small value to be added to the Covariances for numerical stability
        #
        self._n_gaussians = n_gaussians
        self._n_sample_points = n_sample_points
        self._initialization_method = initialization_method
        assert (0 <= initialization_method <= 3)
        self._termination_criterion = termination_criterion
        self._m_step_gaussians_subbatchsize = em_step_gaussians_subbatchsize
        self._m_step_points_subbatchsize = em_step_points_subbatchsize
        self._logger = None
        self._epsvar = eps
        self._eps = None
        self._dtype = dtype

    def set_logging(self, logger: GMLogger = None):
        # Sets logging options
        # Paramters:
        #   logger: GMLogger
        #       GMLogger object to call every iteration
        #
        self._logger = logger

    def generate(self, pcbatch: torch.Tensor, gmbatch: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        # Gets a point cloud batch of size [m,n,3]
        # where m is the batch size and n the point count.
        # If the given logger uses a scaler, the point cloud has to be be given downscaled!
        # It might be given an initial gaussian mixture of
        # size [m,1,g,10] where m is the batch size and g
        # the number of Gaussians.
        # It returns two gaussian mixtures of sizes
        # [m,1,g,10], the first being a mixture with amplitudes as weights
        # the second a mixture where the weights describe the priors.
        # Training parameters have to be set in the other methods
        # of the class

        # Initializations
        self._termination_criterion.reset()

        batch_size = pcbatch.shape[0]
        point_count = pcbatch.shape[1]
        n_sample_points = min(point_count, self._n_sample_points)
        if n_sample_points == -1:
            n_sample_points = point_count
        pcbatch = pcbatch.to(self._dtype).cuda()  # dimension: (bs, np, 3)

        assert (point_count > self._n_gaussians)

        # eps is a small multiple of the identity matrix which is added to the cov-matrizes
        # in order to avoid singularities
        self._eps = (torch.eye(3, 3, dtype=self._dtype) * self._epsvar).view(1, 1, 1, 3, 3) \
            .expand(batch_size, 1, self._n_gaussians, 3, 3).cuda()

        # running defines which batches are still being trained
        running = torch.ones(batch_size, dtype=torch.bool)

        # Initialize mixture data
        if gmbatch is None:
            if self._initialization_method == 0:
                gmbatch = self.initialize_rand1(pcbatch)
            elif self._initialization_method == 1:
                gmbatch = self.initialize_rand2(pcbatch)
            elif self._initialization_method == 2:
                gmbatch = self.initialize_adam1(pcbatch)
            else:
                gmbatch = self.initialize_adam2(pcbatch)
        gm_data = self.TrainingData(batch_size, self._n_gaussians, self._dtype)
        gm_data.set_positions(gm.positions(gmbatch), running)
        gm_data.set_covariances(gm.covariances(gmbatch), running)
        gm_data.set_amplitudes(gm.weights(gmbatch), running)

        iteration = 0

        # last losses. saved so we have losses for gms that are already finished
        last_losses = torch.ones(batch_size, dtype=self._dtype).cuda()
        while True:
            iteration += 1

            # Sample points for this iteration
            if n_sample_points < point_count:
                sample_points = data_loading.sample(pcbatch, n_sample_points)
            else:
                sample_points = pcbatch
            points_rep = sample_points.unsqueeze(1).unsqueeze(3) \
                .expand(batch_size, 1, n_sample_points, self._n_gaussians, 3)

            # Expectation: Calculates responsibilities and current losses
            responsibilities, losses = self._expectation(points_rep, gm_data, running, last_losses)
            last_losses = losses

            # Log Loss (before changing the values in the maximization step,
            # so basically we use the logg of the previous iteration)
            loss = losses.sum()

            assert not torch.isnan(loss).any()
            if self._logger:
                self._logger.log(iteration - 1, losses, gm_data.pack_mixture(), running)

            # If in the previous iteration we already reached the termination criteration, stop now
            # and do not perform the maximization step
            running = self._termination_criterion.may_continue(iteration - 1, losses)
            if not running.any():
                break

            # Maximization -> update GM-data
            self._maximization(points_rep, responsibilities, gm_data, running)

        # Create final mixtures
        final_gm = gm_data.pack_mixture().float()
        final_gmm = gm_data.pack_mixture_model().float()

        # Gaussian-Weights might be set to zero. This prints for how many Gs this is the case
        print("EM: # of invalid Gaussians: ", torch.sum(gm_data.get_priors() == 0).item())

        return final_gm, final_gmm

    def _expectation(self, points: torch.Tensor, gm_data, running: torch.Tensor, losses: torch.Tensor = None) -> \
            (torch.Tensor, torch.Tensor):
        # This performs the Expectation step of the EM Algorithm. This calculates 1) the responsibilities.
        # So the probabilities, how likely each point belongs to each gaussian and 2) the overall Log-Likelihood
        # of this GM given the point cloud.
        # The calculations are performed numerically stable in Log-Space!
        # Per default, all points and Gaussians are processed at once.
        # However, by setting em_step_gaussians_subbatchsize and em_step_points_subbatchsize in the constructor,
        # this can be split into several processings to save memory.
        # Parameters:
        #   points: torch.Tensor of shape (batch_size, 1, n_points, 1, 3)
        #       This is a expansion of the (sampled) point cloud
        #   gm_data: TrainingData
        #       The current GM-object
        #   running: torch.Tensor of shape (batch_size), dtype=bool
        #       Gives information on which GMs are still running (true) or finished (false). Only relevant GMs will
        #       be considered. Responsibilities of GMs that are finished will be zero.
        #   losses: torch.Tensor of shape (batch_size)
        #       Losses from last iteration. Will be returned for all the GMs that are not updated anymore
        #       Can also be None, then 1 will be returned for inactive GMs.
        # Returns:
        #   responsibilities: torch.Tensor of shape (batch_size, 1, n_points, n_gaussians)
        #   losses: torch.Tensor of shape (batch_size): Negative Log-Likelihood for each GM

        batch_size = points.shape[0]
        running_batch_size = running.sum()
        n_sample_points = points.shape[2]

        # This uses the fact that
        # log(a * exp(-0.5 * M(x))) = log(a) + log(exp(-0.5 * M(x))) = log(a) - 0.5 * M(x)

        gauss_subbatch_size = self._m_step_gaussians_subbatchsize
        if gauss_subbatch_size < 1:
            gauss_subbatch_size = self._n_gaussians
        point_subbatch_size = self._m_step_points_subbatchsize
        if point_subbatch_size < 1:
            point_subbatch_size = n_sample_points

        likelihood_log = \
            torch.zeros(running_batch_size, 1, n_sample_points, self._n_gaussians, dtype=self._dtype).cuda()
        gmpos = gm_data.get_positions()
        gmicov = gm_data.get_inversed_covariances()
        gmloga = gm_data.get_logarithmized_amplitudes()

        for j_start in range(0, self._n_gaussians, gauss_subbatch_size):
            j_end = j_start + gauss_subbatch_size
            actual_gauss_subbatch_size = min(self._n_gaussians, j_end) - j_start
            for i_start in range(0, n_sample_points, point_subbatch_size):
                i_end = i_start + point_subbatch_size
                actual_point_subbatch_size = min(n_sample_points, i_end) - i_start
                points_rep = points[running, :, i_start:i_end, :, :]\
                    .expand(running_batch_size, 1, actual_point_subbatch_size, actual_gauss_subbatch_size, 3)
                # GM-Positions, expanded for each PC point. shape: (bs, 1, np, ng, 3)
                gmpositions_rep = gmpos[running, :, j_start:j_end].unsqueeze(2)\
                    .expand(running_batch_size, 1, actual_point_subbatch_size, actual_gauss_subbatch_size, 3)
                # GM-Inverse Covariances, expanded for each PC point. shape: (bs, 1, np, ng, 3, 3)
                gmicovs_rep = gmicov[running, :, j_start:j_end].unsqueeze(2)\
                    .expand(running_batch_size, 1, actual_point_subbatch_size, actual_gauss_subbatch_size, 3, 3)
                # Tensor of {PC-point minus GM-position}-vectors. shape: (bs, 1, np, ng, 3, 1)
                grelpos = (points_rep - gmpositions_rep).unsqueeze(5)
                # Tensor of 0.5 times the Mahalanobis distances of PC points to Gaussians. shape: (bs, 1, np, ng)
                expvalues = 0.5 * \
                    torch.matmul(grelpos.transpose(-2, -1), torch.matmul(gmicovs_rep, grelpos)).squeeze(5).squeeze(4)
                # Logarithmized GM-Priors, expanded for each PC point. shape: (bs, 1, np, ng)
                gmpriors_log_rep = gmloga[running, :, j_start:j_end].unsqueeze(2) \
                    .expand(running_batch_size, 1, actual_point_subbatch_size, actual_gauss_subbatch_size)
                # The logarithmized likelihoods of each point for each gaussian. shape: (bs, 1, np, ng)
                likelihood_log[:, :, i_start:i_end, j_start:j_end] = gmpriors_log_rep - expvalues

        # Logarithmized Likelihood for each point given the GM. shape: (bs, 1, np, ng)
        llh_sum = torch.logsumexp(likelihood_log, dim=3, keepdim=True)
        # Logarithmized Mean Likelihood for all points. shape: (bs)
        if losses is None:
            losses = torch.zeros(batch_size, dtype=self._dtype).cuda()
        losses[running] = -llh_sum.mean(dim=2).view(running_batch_size)
        # Calculating responsibilities
        responsibilities = torch.zeros(batch_size, 1, n_sample_points, self._n_gaussians, dtype=self._dtype).cuda()
        responsibilities[running] = torch.exp(likelihood_log - llh_sum)
        # Calculating responsibilities and returning them and the mean loglikelihoods
        return responsibilities, losses

    def _maximization(self, points_rep: torch.Tensor, responsibilities: torch.Tensor, gm_data, running: torch.Tensor):
        # This performs the Maximization step of the EM Algorithm.
        # Updates the GM-Model given the responsibilities which resulted from the E-Step.
        # Per default, all points and Gaussians are processed at once.
        # However, by setting em_step_gaussians_subbatchsize and em_step_points_subbatchsize in the constructor,
        # this can be split into several processings to save memory.
        # Parameters:
        #   points_rep: torch.Tensor of shape (batch_size, 1, n_points, n_gaussians, 3)
        #       This is a expansion of the (sampled) point cloud, repeated n_gaussian times along dimension 4
        #   responsibilities: torch.Tensor of shape (batch_size, 1, n_points, n_gaussians)
        #       This is the result of the E-step.
        #   gm_data: TrainingData
        #       The current GM-object (will be changed)
        #   running: torch.Tensor of shape (batch_size), dtype=bool
        #       Gives information on which GMs are still running (true) or finished (false).

        n_sample_points = points_rep.shape[2]
        n_running = running.sum()
        gauss_subbatch_size = self._m_step_gaussians_subbatchsize
        if gauss_subbatch_size < 1:
            gauss_subbatch_size = self._n_gaussians
        point_subbatch_size = self._m_step_points_subbatchsize
        if point_subbatch_size < 1:
            point_subbatch_size = n_sample_points

        new_positions = gm_data.get_positions()[running].clone()
        new_covariances = gm_data.get_covariances()[running].clone()
        new_priors = gm_data.get_priors()[running].clone()

        # Iterate over Gauss-Subbatches
        for j_start in range(0, self._n_gaussians, gauss_subbatch_size):
            j_end = j_start + gauss_subbatch_size
            actual_gauss_subbatch_size = min(self._n_gaussians, j_end) - j_start
            # Initialize T-Variables for these Gaussians, will be filled in the upcoming loop
            # Positions/Covariances/Priors are calculated from these (see Eckart-Paper)
            t_0 = torch.zeros(n_running, 1, actual_gauss_subbatch_size, dtype=self._dtype).cuda()
            t_1 = torch.zeros(n_running, 1, actual_gauss_subbatch_size, 3, dtype=self._dtype).cuda()
            t_2 = torch.zeros(n_running, 1, actual_gauss_subbatch_size, 3, 3, dtype=self._dtype).cuda()

            # Iterate over Point-Subbatches
            for i_start in range(0, n_sample_points, point_subbatch_size):
                i_end = i_start + point_subbatch_size
                relevant_responsibilities = responsibilities[running, :, i_start:i_end, j_start:j_end]
                # actual_point_subbatch_size = relevant_responsibilities.shape[2]
                relevant_points = points_rep[running, :, i_start:i_end, j_start:j_end, :]
                matrices_from_points = relevant_points.unsqueeze(5) * relevant_points.unsqueeze(5).transpose(-1, -2)
                # Fill T-Variables      # t_2 shape: (1, 1, J, 3, 3)
                t_2 += (matrices_from_points * relevant_responsibilities.unsqueeze(4).unsqueeze(5)).sum(dim=2)
                t_0 += relevant_responsibilities.sum(dim=2)  # shape: (1, 1, J)
                t_1 += (relevant_points * relevant_responsibilities.unsqueeze(4)) \
                    .sum(dim=2)  # shape: (1, 1, J, 3)
                del matrices_from_points, relevant_points

            new_positions[:, :, j_start:j_end] = t_1 / t_0.unsqueeze(3)  # (bs, 1, ng, 3)
            new_covariances[:, :, j_start:j_end] = t_2 / t_0.unsqueeze(3).unsqueeze(4) - \
                (new_positions[:, :, j_start:j_end].unsqueeze(4) * new_positions[:, :, j_start:j_end].unsqueeze(4)
                 .transpose(-1, -2)) + self._eps[running]
            new_priors[:, :, j_start:j_end] = t_0 / n_sample_points
            del t_0, t_1, t_2

        # Handling of invalid Gaussians! If all responsibilities of a Gaussian are zero, the previous code will
        # set the prior of it to zero and the covariances and positions to NaN
        # To avoid NaNs, we will then replace those invalid values with 0 (pos) and eps (cov).
        new_positions[new_priors == 0] = torch.tensor([0.0, 0.0, 0.0], dtype=self._dtype).cuda()
        new_covariances[new_priors == 0] = self._eps[0, 0, 0, :, :]

        # Update GMData
        gm_data.set_positions(new_positions, running)
        gm_data.set_covariances(new_covariances, running)
        gm_data.set_priors(new_priors, running)

        assert not torch.isnan(gm_data.get_logarithmized_amplitudes()).any(),\
            "Numerical Issues! Consider increasing precision."

    def initialize_rand1(self, pcbatch: torch.Tensor) -> torch.Tensor:
        # Creates a new initial Gaussian Mixture (batch) for a given point cloud (batch).
        # The initialization is done according to McLachlan and Peel "Finite Mixture Models" (2000), Chapter 2.12.2
        # The positions are sampled from a normal distribution based on the empirical mean and covariances
        # of the point cloud. The covariances of the Gaussians are equal to the empirical covariances of the
        # point cloud.
        # Parameters:
        #   pcbatch: torch.Tensor(batch_size, n_points, 3)
        batch_size = pcbatch.shape[0]
        point_count = pcbatch.shape[1]

        # Calculate the mean pc position. shape: (bs, 1, 3)
        meanpos = pcbatch.mean(dim=1, keepdim=True)
        # Calcualte (point - meanpoint) pairs. Shape: (bs, np, 3, 1)
        diffs = (pcbatch - meanpos.expand(batch_size, point_count, 3)).unsqueeze(3)
        # Squeeze meanpos -> shape: (bs, 3)
        meanpos = meanpos.squeeze(1)
        # Calculate expected covariance. Shape: (bs, 3, 3)
        meancov = (diffs * diffs.transpose(-1, -2)).mean(dim=[1])
        # Calculated mean prior.
        meanweight = 1.0 / self._n_gaussians

        # Sample positions from Gaussian -> shape: (bs, 1, ng, 3)
        positions = torch.zeros(batch_size, 1, self._n_gaussians, 3).to(self._dtype)
        for i in range(batch_size):
            positions[i, 0, :, :] = torch.tensor(
                numpy.random.multivariate_normal(meanpos[i, :].cpu(), meancov[i, :, :].cpu(), self._n_gaussians)).cuda()
        # Repeat covariances for each Gaussian -> shape: (bs, 1, ng, 3, 3)
        covariances = meancov.view(batch_size, 1, 1, 3, 3).expand(batch_size, 1, self._n_gaussians, 3, 3)
        # Set weight for each Gaussian -> shape: (bs, 1, ng)
        weights = torch.zeros(batch_size, 1, self._n_gaussians).to(self._dtype)
        weights[:, :, :] = meanweight

        # pack gmm-mixture
        return gm.pack_mixture(weights.cuda(), positions.cuda(), covariances.cuda()).to(self._dtype)

    def initialize_rand2(self, pcbatch: torch.Tensor) -> torch.Tensor:
        # Creates a new initial Gaussian Mixture (batch) for a given point cloud (batch).
        # The initialization is done according to McLachlan and Peel "Finite Mixture Models" (2000), Chapter 2.12.2
        # The responsibilities are created somewhat randomly and from these the M step calculates the Gaussians.
        # Parameters:
        #   pcbatch: torch.Tensor(batch_size, n_points, 3)
        batch_size = pcbatch.shape[0]
        point_count = pcbatch.shape[1]
        n_sample_points = min(point_count, self._n_sample_points)
        if n_sample_points < point_count:
            sample_points = data_loading.sample(pcbatch, n_sample_points)
        else:
            sample_points = pcbatch

        if self._eps is None:
            self._eps = (torch.eye(3, 3, dtype=self._dtype) * self._epsvar).view(1, 1, 1, 3, 3) \
                .expand(batch_size, 1, self._n_gaussians, 3, 3).cuda()

        assignments = torch.randint(low=0, high=self._n_gaussians, size=(batch_size * n_sample_points,))
        point_indizes = torch.arange(0, n_sample_points).repeat(batch_size)
        batch_indizes = torch.arange(0, batch_size).repeat(n_sample_points, 1).transpose(-1, -2).reshape(-1)
        responsibilities = torch.zeros(batch_size, n_sample_points, self._n_gaussians).cuda()
        responsibilities[batch_indizes, point_indizes, assignments] = 1
        responsibilities = responsibilities.unsqueeze(1).to(self._dtype)

        gm_data = self.TrainingData(batch_size, self._n_gaussians, self._dtype)
        sp_rep = sample_points.unsqueeze(1).unsqueeze(3).expand(batch_size, 1, n_sample_points, self._n_gaussians, 3)
        self._maximization(sp_rep, responsibilities, gm_data, torch.ones(batch_size, dtype=torch.bool))
        return gm_data.pack_mixture_model().cuda().to(self._dtype)

    def initialize_adam1(self, pcbatch: torch.Tensor) -> torch.Tensor:
        # Creates a new initial Gaussian Mixture (batch) for a given point cloud (batch).
        # The initialization is done according to Adam's method.
        # Furthest Point Sampling for mean selection, then assigning each point to the closest mean, then performing
        # an M step.
        # Parameters:
        #   pcbatch: torch.Tensor(batch_size, n_points, 3)
        batch_size = pcbatch.shape[0]

        sampled = furthest_point_sampling.apply(pcbatch.float(), self._n_gaussians).to(torch.long).reshape(-1)
        batch_indizes = torch.arange(0, batch_size).repeat(self._n_gaussians, 1).transpose(-1, -2).reshape(-1)
        gmpositions = pcbatch[batch_indizes, sampled, :].view(batch_size, 1, self._n_gaussians, 3)
        # Achtung! Das gibt wohl eher die Indizes zurück
        gmcovariances = torch.zeros(batch_size, 1, self._n_gaussians, 3, 3).to(self._dtype).cuda()
        gmcovariances[:, :, :] = torch.eye(3, dtype=self._dtype).cuda()
        gmpriors = torch.zeros(batch_size, 1, self._n_gaussians).to(self._dtype).cuda()
        gmpriors[:, :, :] = 1 / self._n_gaussians

        return gm.pack_mixture(gmpriors, gmpositions, gmcovariances)

    def initialize_adam2(self, pcbatch: torch.Tensor) -> torch.Tensor:
        # Creates a new initial Gaussian Mixture (batch) for a given point cloud (batch).
        # The initialization is done according to Adam's method.
        # Furthest Point Sampling for mean selection, then assigning each point to the closest mean, then performing
        # an M step
        # Parameters:
        #   pcbatch: torch.Tensor(batch_size, n_points, 3)
        batch_size = pcbatch.shape[0]
        point_count = pcbatch.shape[1]

        n_sample_points = min(point_count, self._n_sample_points)
        if n_sample_points < point_count:
            sample_points = data_loading.sample(pcbatch, n_sample_points)
        else:
            sample_points = pcbatch

        if self._eps is None:
            self._eps = (torch.eye(3, 3, dtype=self._dtype) * self._epsvar).view(1, 1, 1, 3, 3) \
                .expand(batch_size, 1, self._n_gaussians, 3, 3).cuda()

        mix = self.initialize_adam1(pcbatch)

        gm_data = self.TrainingData(batch_size, self._n_gaussians, self._dtype)
        running = torch.ones(batch_size, dtype=torch.bool)
        gm_data.set_positions(gm.positions(mix), running)
        gm_data.set_covariances(gm.covariances(mix), running)
        gm_data.set_priors(gm.weights(mix), running)

        sp_rep = sample_points.unsqueeze(1).unsqueeze(3).expand(batch_size, 1, n_sample_points, self._n_gaussians, 3)

        responsibilities, llh = self._expectation(sp_rep, gm_data, running)

        # resp dimension: (batch_size, 1, n_points, n_gaussians)
        # let's find the maximum per gaussian
        assignments = responsibilities.argmax(dim=3).view(-1)
        point_indizes = torch.arange(0, n_sample_points).repeat(batch_size)
        batch_indizes = torch.arange(0, batch_size).repeat(n_sample_points, 1).transpose(-1, -2).reshape(-1)
        assignedresps = torch.zeros(batch_size, n_sample_points, self._n_gaussians).cuda()
        assignedresps[batch_indizes, point_indizes, assignments] = 1
        assignedresps = assignedresps.unsqueeze(1).to(self._dtype)

        self._maximization(sp_rep, assignedresps, gm_data, running)
        return gm_data.pack_mixture_model().cuda().to(self._dtype)

    class TrainingData:
        # Helper class. Capsules all relevant training data of the current GM batch.
        # positions, covariances and priors are stored as-is and can be set.
        # inversed covariances are calcualted whenever covariances are set.
        # amplitudes are calculated from priors (or vice versa).
        # Note that priors or amplitudes should always be set after the covariances are set,
        # otherwise the conversion is not correct anymore.

        def __init__(self, batch_size, n_gaussians, dtype):
            self._positions = torch.zeros(batch_size, 1, n_gaussians, 3, dtype=dtype).cuda()
            self._logamplitudes = torch.zeros(batch_size, 1, n_gaussians, dtype=dtype).cuda()
            self._priors = torch.zeros(batch_size, 1, n_gaussians, dtype=dtype).cuda()
            self._covariances = torch.zeros(batch_size, 1, n_gaussians, 3, 3, dtype=dtype).cuda()
            self._inversed_covariances = torch.zeros(batch_size, 1, n_gaussians, 3, 3, dtype=dtype).cuda()

        def set_positions(self, positions, running):
            # running indicates which batch entries should be replaced
            self._positions[running] = positions

        def set_covariances(self, covariances, running):
            # running indicates which batch entries should be replaced
            relcovs = ~torch.isnan(covariances.det().sqrt())    # ToDo: Think about this approach
            runningcovs = self._covariances[running]
            runningcovs[relcovs] = covariances[relcovs]
            self._covariances[running] = runningcovs
            runningicovs = self._inversed_covariances[running]
            runningicovs[relcovs] = covariances[relcovs].inverse().contiguous()
            self._inversed_covariances[running] = runningicovs

        def set_amplitudes(self, amplitudes, running):
            # running indicates which batch entries should be replaced
            self._logamplitudes[running] = torch.log(amplitudes)
            self._priors[running] = amplitudes * (self._covariances[running].det().sqrt() * 15.74960995)

        def set_priors(self, priors, running):
            # running indicates which batch entries should be replaced
            self._priors[running] = priors
            self._logamplitudes[running] = torch.log(priors) - \
                0.5 * torch.log(self._covariances[running].det()) - 2.7568156719207764

        def get_positions(self):
            return self._positions

        def get_covariances(self):
            return self._covariances

        def get_inversed_covariances(self):
            return self._inversed_covariances

        def get_priors(self):
            return self._priors

        def get_amplitudes(self):
            return torch.exp(self._logamplitudes)

        def get_logarithmized_amplitudes(self):
            return self._logamplitudes

        def pack_mixture(self):
            return gm.pack_mixture(torch.exp(self._logamplitudes), self._positions, self._covariances)

        def pack_mixture_model(self):
            return gm.pack_mixture(self._priors, self._positions, self._covariances)
