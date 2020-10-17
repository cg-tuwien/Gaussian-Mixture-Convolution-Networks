from prototype_pcfitting import GMMGenerator, GMLogger, data_loading
from prototype_pcfitting import TerminationCriterion, MaxIterationTerminationCriterion
import torch
import gmc.mixture as gm
import numpy


class EMGenerator(GMMGenerator):
    # GMM Generator using simple Expectation Maximization (numerically stable)

    def __init__(self,
                 n_gaussians: int,
                 n_sample_points: int,
                 termination_criterion: TerminationCriterion = MaxIterationTerminationCriterion(100),
                 dtype: torch.dtype = torch.float32):
        # Constructor. Creates a new EMGenerator.
        # Parameters:
        #   n_gaussians: int
        #       Number of components this Generator should create.
        #       This should always be set correctly, also when this is used for refining.
        #   n_sample_points: int
        #       Number of points to use each iteration
        #   termination_criteration: TerminationCriterion
        #       Defining when to terminate. Default: After 100 Iterations.
        #       As this algorithm works on batches, the common batch loss is given to the termination criterion
        #       (We could implement saving of the best result in order to avoid moving out of optima)
        #   dtype: torch.dtype
        #       In which data type (precision) the operations should be performed. The final gmm is always
        #       converted to float32 though. Default: torch.float32
        #
        self._n_gaussians = n_gaussians
        self._n_sample_points = n_sample_points
        self._termination_criterion = termination_criterion
        self._logger = None
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
        pcbatch = pcbatch.to(self._dtype).cuda()  # dimension: (bs, np, 3)

        assert (point_count > self._n_gaussians)

        # Initialize mixture data
        if gmbatch is None:
            gmbatch = self.initialize(pcbatch)
        gm_data = self.TrainingData()
        gm_data.set_positions(gm.positions(gmbatch))
        gm_data.set_covariances(gm.covariances(gmbatch))
        gm_data.set_amplitudes(gm.weights(gmbatch))

        iteration = 0

        # eps is a small multiple of the identity matrix which is added to the cov-matrizes
        # in order to avoid singularities
        self._eps = (torch.eye(3, 3, dtype=self._dtype) * 1e-6).view(1, 1, 1, 3, 3) \
            .expand(batch_size, 1, self._n_gaussians, 3, 3).cuda()

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
            responsibilities, losses = self.expectation(points_rep, gm_data)

            # Log Loss (before changing the values in the maximization step,
            # so basically we use the logg of the previous iteration)
            loss = losses.sum()
            if self._logger:
                self._logger.log(iteration - 1, losses, gm_data.pack_mixture())

            # If in the previous iteration we already reached the termination criteration, stop now
            # and do not perform the maximization step
            if not self._termination_criterion.may_continue(iteration - 1, loss.item()):
                break

            # Maximization -> update GM-data
            self.maximization(points_rep, responsibilities, gm_data)

        # Create final mixtures
        final_gm = gm_data.pack_mixture().float()
        final_gmm = gm.pack_mixture(gm_data.get_priors(), gm_data.get_positions(), gm_data.get_covariances())

        # Gaussian-Weights might be set to zero. This prints for how many Gs this is the case
        print("EM: # of invalid Gaussians: ", torch.sum(gm_data.get_priors() == 0).item())

        return final_gm, final_gmm

    def expectation(self, points_rep: torch.Tensor, gm_data) -> (torch.Tensor, torch.Tensor):
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
            .unsqueeze(2).expand(batch_size, 1, n_sample_points, self._n_gaussians, 3)
        # GM-Inverse Covariances, expanded for each PC point. shape: (bs, 1, np, ng, 3, 3)
        gmicovs_rep = gm_data.get_inversed_covariances() \
            .unsqueeze(2).expand(batch_size, 1, n_sample_points, self._n_gaussians, 3, 3)
        # Tensor of {PC-point minus GM-position}-vectors. shape: (bs, 1, np, ng, 3, 1)
        grelpos = (points_rep - gmpositions_rep).unsqueeze(5)
        # Tensor of 0.5 times the Mahalanobis distances of PC points to Gaussians. shape: (bs, 1, np, ng)
        expvalues = 0.5 * \
            torch.matmul(grelpos.transpose(-2, -1), torch.matmul(gmicovs_rep, grelpos)).squeeze(5).squeeze(4)
        # Logarithmized GM-Priors, expanded for each PC point. shape: (bs, 1, np, ng)
        gmpriors_log_rep = \
            torch.log(gm_data.get_amplitudes().unsqueeze(2).expand(batch_size, 1, n_sample_points, self._n_gaussians))
        # The logarithmized likelihoods of each point for each gaussian. shape: (bs, 1, np, ng)
        likelihood_log = gmpriors_log_rep - expvalues
        # Logarithmized Likelihood for each point given the GM. shape: (bs, 1, np, ng)
        llh_sum = torch.logsumexp(likelihood_log, dim=3, keepdim=True)
        # Logarithmized Mean Likelihood for all points. shape: (bs)
        losses = -llh_sum.mean(dim=2).view(batch_size)
        # Calculating responsibilities and returning them and the mean loglikelihoods
        return torch.exp(likelihood_log - llh_sum), losses

    def maximization(self, points_rep: torch.Tensor, responsibilities: torch.Tensor, gm_data):
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
        new_positions_rep = new_positions.expand(batch_size, 1, n_sample_points, self._n_gaussians, 3)
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

    def initialize(self, pcbatch: torch.Tensor) -> torch.Tensor:
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

    class TrainingData:
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
