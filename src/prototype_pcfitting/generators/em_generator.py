from prototype_pcfitting import GMMGenerator, GMLogger, data_loading
from prototype_pcfitting import TerminationCriterion, MaxIterationTerminationCriterion
from gmc.cpp.extensions.furthest_point_sampling import furthest_point_sampling
import torch
import gmc.mixture as gm
import numpy
from sklearn.cluster import KMeans
from .gmm_initializer import GMMInitializer
from .em_tools import EMTools


class EMGenerator(GMMGenerator):
    # GMM Generator using simple Expectation Maximization (numerically stable)

    def __init__(self,
                 n_gaussians: int,
                 n_sample_points: int = -1,
                 termination_criterion: TerminationCriterion = MaxIterationTerminationCriterion(100),
                 initialization_method: str = 'randnormpos',
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
        #   initialization_method: string
        #       Defines which initialization to use. All options from GMMInitializer are available:
        #       'randnormpos' or 'rand1' = Random by sample mean and cov
        #       'randresp' or 'rand2' = Random responsibilities,
        #       'fsp' or 'adam1' = furthest point sampling,
        #       'fspmax' or 'adam2' = furthest point sampling, artifical responsibilities and m-step,
        #       'kmeans-full' = Full kmeans,
        #       'kmeans-fast' or 'kmeans' = Fast kmeans
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
        self._termination_criterion = termination_criterion
        self._em_step_gaussians_subbatchsize = em_step_gaussians_subbatchsize
        self._em_step_points_subbatchsize = em_step_points_subbatchsize
        self._logger = None
        self._epsvar = eps
        self._dtype = dtype
        self._initializer = GMMInitializer(em_step_gaussians_subbatchsize, em_step_points_subbatchsize, dtype, eps)

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
        eps = (torch.eye(3, 3, dtype=self._dtype) * self._epsvar).view(1, 1, 1, 3, 3) \
            .expand(batch_size, 1, self._n_gaussians, 3, 3).cuda()

        # running defines which batches are still being trained
        running = torch.ones(batch_size, dtype=torch.bool)

        # Initialize mixture data
        if gmbatch is None:
            gmbatch = self._initializer.initialize_by_method_name(self._initialization_method, pcbatch,
                                                                  self._n_gaussians, self._n_sample_points)
        gm_data = EMTools.TrainingData(batch_size, self._n_gaussians, self._dtype)
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
            responsibilities, losses = EMTools.expectation(points_rep, gm_data, self._n_gaussians, running,
                                                           self._em_step_gaussians_subbatchsize,
                                                           self._em_step_points_subbatchsize, last_losses)
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
            EMTools.maximization(points_rep, responsibilities, gm_data, running, eps,
                                 self._em_step_gaussians_subbatchsize, self._em_step_points_subbatchsize)

        # Create final mixtures
        final_gm = gm_data.pack_mixture().float()
        final_gmm = gm_data.pack_mixture_model().float()

        # Gaussian-Weights might be set to zero. This prints for how many Gs this is the case
        print("EM: # of invalid Gaussians: ", torch.sum(gm_data.get_priors() == 0).item())

        return final_gm, final_gmm


