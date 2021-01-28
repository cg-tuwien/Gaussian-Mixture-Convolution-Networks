from prototype_pcfitting import GMMGenerator, GMLogger, data_loading
from prototype_pcfitting import TerminationCriterion, MaxIterationTerminationCriterion
import torch
import gmc.mixture as gm
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
                 use_noise_cluster: bool = False,
                 dtype: torch.dtype = torch.float32,
                 eps: float = 1e-7,
                 eps_is_relative: bool = True):
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
        #       'fps' or 'adam1' = furthest point sampling,
        #       'fpsmax' or 'adam2' = furthest point sampling, artifical responsibilities and m-step,
        #       'kmeans-full' = Full kmeans,
        #       'kmeans-fast' or 'kmeans' = Fast kmeans
        #   em_step_gaussian_subbatchsize: int
        #       How many Gaussian Sub-Mixtures should be processed in the E- and M-Step at once
        #       -1 means all Gaussians (default)
        #   em_step_points_subbatchsize: int
        #       How many points should be processed in the E- and M-Step at once
        #       -1 means all Points (default)
        #   use_noise_cluster: bool
        #       If true, a noise cluster is used, meaning a weighted uniform distribution over the boudning box
        #   dtype: torch.dtype
        #       In which data type (precision) the operations should be performed. Default: torch.float32
        #   eps: float
        #       Small value to be added to the covariances' diagonals for numerical stability
        #   eps_is_relative: bool
        #       If false, eps is added as is to the covariances. If true (default), this eps is relative
        #       to the longest side of the pc's bounding box (recommended for scaling invariance).
        #       eps_abs = eps_rel * (maxextend^2)
        #
        self._n_gaussians = n_gaussians
        self._n_sample_points = n_sample_points
        self._initialization_method = initialization_method
        self._termination_criterion = termination_criterion
        self._em_step_gaussians_subbatchsize = em_step_gaussians_subbatchsize
        self._em_step_points_subbatchsize = em_step_points_subbatchsize
        self._use_noise_cluster = use_noise_cluster
        self._logger = None
        self._epsvar = eps
        if eps < 1e-9:
            print("Warning! Very small eps! Might cause numerical issues!")
        self._eps_is_relative = eps_is_relative
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

        initnoisevalues = None
        if self._use_noise_cluster:
            extends = pcbatch.max(dim=1)[0] - pcbatch.min(dim=1)[0]
            initnoisevalues = 1 / torch.prod(extends, dim=1)

        epsilons = torch.ones(batch_size, dtype=self._dtype, device='cuda') * self._epsvar
        if self._eps_is_relative:
            extends = pcbatch.max(dim=1)[0] - pcbatch.min(dim=1)[0]
            epsilons *= extends.max(dim=1)[0] ** 2
            epsilons[epsilons < 1e-9] = 1e-9

        # eps is a small multiple of the identity matrix which is added to the cov-matrizes
        # in order to avoid singularities
        eps = (torch.eye(3, 3, dtype=self._dtype, device='cuda')).view(1, 1, 1, 3, 3) \
            .expand(batch_size, 1, 1, 3, 3) * epsilons.view(-1, 1, 1, 1, 1)

        # running defines which batches are still being trained
        running = torch.ones(batch_size, dtype=torch.bool)

        # Initialize mixture data
        if gmbatch is None:
            initializer = GMMInitializer(self._em_step_gaussians_subbatchsize, self._em_step_points_subbatchsize,
                                         self._dtype, epsilons)
            gmbatch_init = initializer.initialize_by_method_name(self._initialization_method, pcbatch,
                                                                 self._n_gaussians, self._n_sample_points, None,
                                                                 initnoisevalues)
        else:
            gmbatch_init = gmbatch
        gm_data = EMTools.TrainingData(batch_size, self._n_gaussians, self._dtype, eps)
        gm_data.set_positions(gm.positions(gmbatch_init), running)
        gm_data.set_covariances(gm.covariances(gmbatch_init), running)
        if gmbatch is None:
            gm_data.set_priors(gm.weights(gmbatch_init), running)
        else:
            gm_data.set_amplitudes(gm.weights(gmbatch_init), running)
        if self._use_noise_cluster:
            gm_data.set_noise(1 - gm.weights(gmbatch_init).sum().view(-1), initnoisevalues)

        del epsilons

        iteration = 0

        # last losses. saved so we have losses for gms that are already finished
        last_losses = torch.ones(batch_size, dtype=self._dtype, device='cuda')
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
        final_gm = gm_data.pack_mixture()
        final_gmm = gm_data.pack_mixture_model()

        # Gaussian-Weights might be set to zero. This prints for how many Gs this is the case
        print("EM: # of invalid Gaussians: ", torch.sum(gm_data.get_priors() == 0).item())

        return final_gm, final_gmm
