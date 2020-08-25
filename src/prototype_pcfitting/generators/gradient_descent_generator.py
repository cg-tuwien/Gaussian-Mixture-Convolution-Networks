from prototype_pcfitting import GMMGenerator, GMLogger
from prototype_pcfitting import TerminationCriterion, MaxIterationTerminationCriterion
from prototype_pcfitting.error_functions import LikelihoodLoss
import prototype_pcfitting.pointcloud as pointcloud
import torch
import torch.optim
import gmc.mixture as gm


class GradientDescentGenerator(GMMGenerator):
    # GMM Generator following a Gradient Descent approach

    _device = torch.device('cuda')

    def __init__(self,
                 n_components: int,
                 n_sample_points: int,
                 termination_criterion: TerminationCriterion = MaxIterationTerminationCriterion(3000),
                 learn_rate_pos: float = 1e-3,
                 learn_rate_cov: float = 1e-4,
                 learn_rate_weights: float = 5e-4):
        # Constructor. Creates a new GradientDescentGenerator.
        # Parameters:
        #   n_components: int
        #       Number of components this Generator should create.
        #       This is only used when a new mixture has to be created, not when refining an existing one.
        #   n_sample_points: int
        #       Number of points to consider for calculating the loss
        #   termination_criteration: TerminationCriterion
        #       Defining when to terminate. Default: After 3000 Iterations.
        #       As this algorithm works on batches, the common batch loss is given to the termination criterion
        #       (We could implement saving of the best result in order to avoid moving out of optima)
        #   learn_rate_pos: float
        #       Learning rate for the positions. Default: 1e-3
        #   learn_rate_cov: float
        #       Learning rate for the Cholesky decomposition of the covariances. Default: 1e-4
        #   learn_rate_weights: float
        #       Learning rate for the relative weights. Default: 5e-4
        #
        self._n_components = n_components
        self._n_sample_points = n_sample_points
        self._termination_criterion = termination_criterion
        self._learn_rate_pos = learn_rate_pos
        self._learn_rate_cov = learn_rate_cov
        self._learn_rate_weights = learn_rate_weights
        self._loss = LikelihoodLoss()
        self._logger = None

    def set_logging(self,
                    logger: GMLogger = None):
        # Sets some logging options
        # Paramters:
        #   logger: GMLogger
        #       GMLogger object to call every iteration
        #
        self._logger = logger

    def generate(self, pcbatch: torch.Tensor, gmbatch: torch.Tensor = None) -> torch.Tensor:
        # Gets a point cloud batch of size [m,n,3]
        # where m is the batch size and n the point count.
        # Point cloud is given downscaled (see Scaler)!
        # It might be given an initial gaussian mixture of
        # size [m,1,g,10] where m is the batch size and g
        # the number of Gaussians.
        # It returns a gaussian mixture batch of size
        # [m,1,g,10].
        # Training parameters have to be set in the other methods
        # of the class
        #
        batch_size = pcbatch.shape[0]
        point_count = pcbatch.shape[1]
        pcbatch = pcbatch.to(self._device)

        # Initialize mixture (Important: Assumes intervall [0,1])
        if gmbatch is None:
            gmbatch = gm.generate_random_mixtures(n_batch=batch_size, n_layers=1, n_components=self._n_components,
                                                  n_dims=3, pos_radius=0.5,
                                                  cov_radius=0.01 / (self._n_components ** (1 / 3)),
                                                  weight_min=0, weight_max=1, device=self._device)
            indizes = torch.randperm(point_count)[0:self._n_components]
            positions = pcbatch[:, indizes, :].view(batch_size, 1, self._n_components, 3)
            gmbatch = gm.pack_mixture(gm.weights(gmbatch), positions, gm.covariances(gmbatch))

        # Initialize Training Data
        gm_data = self.TrainingData()
        gm_data.set_positions(gm.positions(gmbatch))
        gm_data.set_covariances(gm.covariances(gmbatch))
        gm_data.set_amplitudes(gm.weights(gmbatch))

        # Initialize Optimizers
        optimiser_pos = torch.optim.RMSprop([gm_data.tr_positions], lr=self._learn_rate_pos, alpha=0.7,
                                                    momentum=0.0)
        optimiser_cov = torch.optim.Adam([gm_data.tr_cov_data], lr=self._learn_rate_cov)
        optimiser_pi = torch.optim.Adam([gm_data.tr_pi_relative], lr=self._learn_rate_weights)

        # Calculate initial loss
        sample_points = pointcloud.sample(pcbatch, self._n_sample_points)
        losses = self._loss.calculate_score(sample_points, gm_data.get_positions(), gm_data.get_covariances(),
                                            gm_data.get_inversed_covariances(), gm_data.get_amplitudes())
        loss = losses.sum()
        iteration = 0

        if self._logger:
            self._logger.log(iteration, losses, gm_data.pack_mixture())

        # Check termination criteria
        while self._termination_criterion.may_continue(iteration, loss.item()):
            iteration += 1
            optimiser_pos.zero_grad()
            optimiser_cov.zero_grad()
            optimiser_pi.zero_grad()

            # Calculate Loss
            sample_points = pointcloud.sample(pcbatch, self._n_sample_points)
            losses = self._loss.calculate_score(sample_points, gm_data.get_positions(), gm_data.get_covariances(),
                                                gm_data.get_inversed_covariances(), gm_data.get_amplitudes())

            # Log
            if self._logger:
                self._logger.log(iteration, losses, gm_data.pack_mixture())

            # Training Step
            loss = losses.sum()
            loss.backward()
            optimiser_pos.step()
            optimiser_cov.step()
            optimiser_pi.step()
            gm_data.update_covariances()
            gm_data.update_amplitudes()

        # Return final mixture
        return gm.pack_mixture(gm_data.get_amplitudes(), gm_data.get_positions(), gm_data.get_covariances())

    class TrainingData:
        # Helper class. Capsules all the training data of the current gm batch
        # Positions are stored as-is
        # Covariance matrices are stored in another format (called the covariance training data), which is based
        # on the Cholesky decomposition. However it only allows covariances of certain sizes, there's a limit
        # to how small the covariances can be.
        # Weights are stored both as GMM-Weights (Prior-Likelihoods) and Amplitudes (everything before the "e^").
        # The training data can be accessed via attributes tr_positions, tr_cov_data and tr_pi_relative
        # Whenever this data changes during training, the functions update_covariances and update_amplitudes
        # have to be called to update the rest of the data.

        def __init__(self):
            # tr-variables are the ones that are trained using backprop + gd
            self.tr_positions = None            # Tensor of shape [m,1,g,3]. The current positions of the Gaussians
            self.tr_cov_data = None             # Tensor of shape [m,1,g,6]. Training data of the covariances
            self.tr_pi_relative = None          # Tensor of shape [m,1,g]. Current non-normalized weights of the GMMs
            self.pi_normalized = None           # Tensor of shape [m,1,g]. Normalized weights of the GMMs
            self.pi_sum = None                  # Tensor of shape [m,1,1]. The sums of all relative weights of the GMMs
            self.covariances = None             # Tensor of shape [m,1,g,3,3]. Covariance matrices of the Gaussians
            self.inversed_covariances = None    # Tensor of shape [m,1,g,3,3]. Inversed covariances of the Gaussians
            self.determinants = None            # Tensor of shape [m,1,g]. Determinants of the Gaussians
            self.amplitudes = None              # Tensor of shape [m,1,g]. Amplitudes of the Gaussians
            self._epsilon = pow(10, -2.6)       # Epsilon for converting training data to covariances and vice versa

        def set_positions(self, positions: torch.Tensor):
            # Initializes the positions. Should only be called once in the beginning.
            self.tr_positions = positions
            self.tr_positions.requires_grad = True

        def set_covariances(self, covariances: torch.Tensor):
            # Initializes the covariances. Should only be called once in the beginning.
            # This creates the covariance training data from the covariances.
            # If there are covariances that are too small, they might be set to bigger ones.
            batch_size = covariances.shape[0]
            n_components = covariances.shape[2]
            cov_factor_mat = torch.cholesky(covariances)
            cov_factor_vec = torch.zeros((batch_size, 1, n_components, 6)).to(GradientDescentGenerator._device)
            cov_factor_vec[:, :, :, 0] = torch.max(cov_factor_mat[:, :, :, 0, 0] - self._epsilon, 0)[0]
            cov_factor_vec[:, :, :, 1] = torch.max(cov_factor_mat[:, :, :, 1, 1] - self._epsilon, 0)[0]
            cov_factor_vec[:, :, :, 2] = torch.max(cov_factor_mat[:, :, :, 2, 2] - self._epsilon, 0)[0]
            cov_factor_vec[:, :, :, 3] = cov_factor_mat[:, :, :, 1, 0]
            cov_factor_vec[:, :, :, 4] = cov_factor_mat[:, :, :, 2, 0]
            cov_factor_vec[:, :, :, 5] = cov_factor_mat[:, :, :, 2, 1]
            self.tr_cov_data = cov_factor_vec
            self.tr_cov_data.requires_grad = True
            self.update_covariances()

        def set_amplitudes(self, amplitudes: torch.Tensor):
            # Initializes the amplitudes. Should only be called once in the beginning
            # Requires amplitudes, not GMM weights. Covariances HAVE TO be set before!
            self.tr_pi_relative = amplitudes.detach() * self.determinants.detach().sqrt() * 15.74960995
            self.tr_pi_relative.requires_grad = True
            self.update_amplitudes()

        def update_covariances(self):
            # Has to be called whenever tr_cov_data is changed, to update covariances, inversed_covariances
            # and determinants
            cov_shape = self.tr_cov_data.shape
            cov_factor_mat_rec = torch.zeros((cov_shape[0], cov_shape[1], cov_shape[2], 3, 3)).to(
                GradientDescentGenerator._device)
            cov_factor_mat_rec[:, :, :, 0, 0] = torch.abs(self.tr_cov_data[:, :, :, 0]) + self._epsilon
            cov_factor_mat_rec[:, :, :, 1, 1] = torch.abs(self.tr_cov_data[:, :, :, 1]) + self._epsilon
            cov_factor_mat_rec[:, :, :, 2, 2] = torch.abs(self.tr_cov_data[:, :, :, 2]) + self._epsilon
            cov_factor_mat_rec[:, :, :, 1, 0] = self.tr_cov_data[:, :, :, 3]
            cov_factor_mat_rec[:, :, :, 2, 0] = self.tr_cov_data[:, :, :, 4]
            cov_factor_mat_rec[:, :, :, 2, 1] = self.tr_cov_data[:, :, :, 5]
            self.covariances = cov_factor_mat_rec @ cov_factor_mat_rec.transpose(-2, -1)
            cov_factor_mat_rec_inv = cov_factor_mat_rec.inverse()
            self.inversed_covariances = cov_factor_mat_rec_inv.transpose(-2, -1) @ cov_factor_mat_rec_inv
            # numerically better way of calculating the determinants
            self.determinants = torch.pow(cov_factor_mat_rec[:, :, :, 0, 0] * cov_factor_mat_rec[:, :, :, 1, 1]
                                          * cov_factor_mat_rec[:, :, :, 2, 2], 2)

        def update_amplitudes(self):
            # Has to be called whenever tr_pi_relative is changed, to update pi_sum, pi_normalized and amplitudes.
            self.pi_sum = self.tr_pi_relative.abs().sum(dim=2).view(-1, 1, 1)
            self.pi_normalized = self.tr_pi_relative.abs() / self.pi_sum
            self.amplitudes = self.pi_normalized / (self.determinants.sqrt() * 15.74960995)

        def get_positions(self):
            return self.tr_positions

        def get_covariances(self):
            return self.covariances

        def get_inversed_covariances(self):
            return self.inversed_covariances

        def get_amplitudes(self):
            return self.amplitudes

        def pack_mixture(self):
            return gm.pack_mixture(self.amplitudes, self.tr_positions, self.covariances)
