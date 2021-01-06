from typing import Tuple

from prototype_pcfitting import GMMGenerator, GMLogger
from prototype_pcfitting import TerminationCriterion, MaxIterationTerminationCriterion
from prototype_pcfitting.error_functions import LikelihoodLoss
from .level_scaler import LevelScaler
import torch
import gmc.mixture as gm
from gmc import mat_tools
import math
from sklearn.cluster import KMeans
from . import EMGenerator
from .gmm_initializer import GMMInitializer


class EckartGeneratorHP(GMMGenerator):
    # GMM Generator using Expectation Sparsification with hard partitioning by Eckart
    # This algorithms first creates a GMM of j Gaussians, then replaces each Gaussian
    # with j new Gaussians, and fits those Sub-GMM to the points which had highest
    # responsibility with that Gaussian.
    # This is faster than classical EM, as instead of n*g responsibilities, only n*j (j << g) responsibilities
    # need to be calculated.

    def __init__(self,
                 n_gaussians_per_node: int,
                 n_levels: int,
                 termination_criterion: TerminationCriterion = MaxIterationTerminationCriterion(20),
                 initialization_method: str = "kmeans",
                 m_step_gaussians_subbatchsize: int = -1,
                 m_step_points_subbatchsize: int = -1,
                 use_scaling: bool = False,
                 scaling_interval: Tuple[float, float] = (0.0, 1.0),
                 dtype: torch.dtype = torch.float64,
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
        #   initialization_method: string
        #       Defines which initialization to use. All options from GMMInitializer are available:
        #           'randnormpos' or 'rand1' = Random by sample mean and cov
        #           'randresp' or 'rand2' = Random responsibilities (NOT RECOMMENDED)
        #           'fps' or 'adam1' = furthest point sampling (NOT RECOMMENDED)
        #           'fpsmax' or 'adam2' = furthest point sampling, artifical responsibilities and m-step,
        #           'kmeans-full' = Full kmeans (NOT RECOMMENDED)
        #           'kmeans-fast' or 'kmeans' = Fast kmeans
        #       Plus some additional methods:
        #           'large-bb': Initialize GMMs on corners of large bounding box of points (with same side lengths)
        #               Warning: This initialization method is deprecated and currently only works with scaling to [0,1]
        #           'tight-bb': Initialize GMMs on corners of tight bounding box of points (different side lengths)
        #           'eigen': Use Eigen vector decomposition to determine initial positions
        #   m_step_gaussian_subbatchsize: int
        #       How many Gaussian Sub-Mixtures should be processed in the M-Step at once (see _maximization)
        #       -1 means all Gaussians (default)
        #   m_step_points_subbatchsize: int
        #       How many points should be processed in the M-Step at once (see _maximization)
        #       -1 means all Points (default)
        #   use_scaling: bool
        #       If each Sub-GM should temporarily be scaled to the scaling_interval (might have an effect on accuracy)
        #   scaling_interval: Tuple[float, float]
        #       The interval each Sub-GM should be scaled to if use_scaling is True
        #   dtype: torch.dtype
        #       In which data type (precision) the operations should be performed. Default: float32
        #   eps: float
        #       Small value to be added to the Covariances for numerical stability
        self._n_gaussians_per_node = n_gaussians_per_node
        self._n_levels = n_levels
        self._initialization_method = initialization_method
        self._termination_criterion = termination_criterion
        self._m_step_gaussians_subbatchsize = m_step_gaussians_subbatchsize
        self._m_step_points_subbatchsize = m_step_points_subbatchsize
        self._use_scaling = use_scaling
        self._scaling_interval = scaling_interval
        self._dtype = dtype
        self._logger = None
        self._eps = (torch.eye(3, 3, dtype=self._dtype, device='cuda') * eps).view(1, 1, 1, 3, 3)
        self._gmminitializer = GMMInitializer(m_step_gaussians_subbatchsize, m_step_points_subbatchsize, dtype, eps)

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
        assert (self._n_levels > 0), "Levels must be > 0"

        batch_size = pcbatch.shape[0]
        assert (batch_size is 1), "EckartGenerator currently does not support batchsizes > 1"
        point_count = pcbatch.shape[1]
        pcbatch = pcbatch.to(self._dtype).cuda()

        # parent_per_point (1,n) identifies which gaussian in the previous layer this point is assigned to
        parent_per_point = torch.zeros(1, point_count, dtype=torch.long, device='cuda')
        # the 0th layer, only has one (fictional) Gaussian, whose index is assumed to be 0
        parent_per_point[:, :] = 0

        llh_loss_calc = LikelihoodLoss()
        mixture = None

        finished_gaussians = torch.tensor([], dtype=torch.long, device='cuda')
        # finished_subgmms: Gaussians that will not be expanded anymore, from each level
        finished_subgmms = []
        # weights of the parents of each point. initialized with one (fictional 0th layer has only one Gaussian)
        parentweights = torch.ones(1, 1, self._n_gaussians_per_node, dtype=self._dtype, device='cuda')

        absiteration = 0  # Iteration index overall
        for level in range(self._n_levels):
            print("Level: ", level)
            parentcount_for_level = self._n_gaussians_per_node ** level  # How many parents the current level has
            relevant_parents = torch.arange(0, parentcount_for_level, device='cuda')  # (parentcount)

            # Scaler, to scale down the sub-pointclouds, and up the resulting sub-gms
            scaler = LevelScaler(active=self._use_scaling, interval=self._scaling_interval)
            scaler.set_pointcloud(pcbatch, parent_per_point, parentcount_for_level)
            points_scaled = scaler.scale_pc(pcbatch)  # (1, np, 3)

            # Initialize GMs
            if self._initialization_method == 'large-bb':
                gm_data = self._initialize_gms_on_unit_cube(relevant_parents, parentweights)
            elif self._initialization_method == 'tight-bb':
                bbs = self._extract_bbs(points_scaled, relevant_parents, parent_per_point)
                gm_data = self._initialize_gms_on_bounding_box(bbs, relevant_parents, parentweights, finished_gaussians)
            elif self._initialization_method == 'eigen':
                gm_data = self._initialize_on_eigen_vectors(relevant_parents, parentweights, parent_per_point, points_scaled)
            else:
                gm_data = self._initialize_per_subgm(relevant_parents, parentweights, parent_per_point, points_scaled)

            self._termination_criterion.reset()

            iteration = 0  # Iteration index for this level
            # EM-Loop
            while True:
                iteration += 1
                absiteration += 1

                points = points_scaled.unsqueeze(1).unsqueeze(3)  # (1, 1, np, 1, 3)

                # E-Step
                responsibilities = self._expectation(points, gm_data, parent_per_point)
                # Calculate Mixture and Loss
                mixture = gm_data.pack_scaled_up_mixture(scaler)
                mixture = self._construct_full_gm(mixture, finished_subgmms)
                loss = llh_loss_calc.calculate_score_packed(pcbatch, mixture)

                if self._logger:
                    self._logger.log(absiteration - 1, loss, mixture)
                    del mixture

                if not self._termination_criterion.may_continue(iteration - 1, loss.view(-1)).item():
                    # update parent_per_point for next level from responsibilities
                    new_parent_indices = responsibilities.argmax(dim=3)
                    parent_per_point = new_parent_indices.view(1, -1)
                    break

                # M-Step
                self._maximization(points, responsibilities, gm_data)

            all_new_parents = torch.tensor(range(len(gm_data)), device='cuda').view(-1, 1, 1)
            finished_gaussians = ((parent_per_point.eq(all_new_parents)).sum(2) == 0).nonzero(as_tuple=False)[:, 0]
            mixture = gm_data.pack_scaled_up_mixture(scaler)
            if level + 1 != self._n_levels:
                finished_subgmms.append(mixture[:, :, finished_gaussians])
            # add gm to hierarchy
            gm_data.scale_up(scaler)
            # update parentweights
            parentweights = gm_data.get_premultiplied_priors().repeat(1, 1, self._n_gaussians_per_node, 1)\
                .transpose(-1, -2).reshape(1, 1, -1)

        # Calculate final GMs
        res_gm = self._construct_full_gm(mixture, finished_subgmms)
        res_gmm = gm.convert_amplitudes_to_priors(res_gm)

        print("Final Loss: ", LikelihoodLoss().calculate_score_packed(pcbatch, res_gm).item())
        # print("EckartHP: # of invalid Gaussians: ", torch.sum(gm.weights(res_gmm).eq(0)).item())

        return res_gm, res_gmm

    def _initialize_gms_on_unit_cube(self, relevant_parents, parentweights):
        print("WARNING! THIS INITIALIZATION METHOD IS DEPRECATED!")
        # Initializes new GMs, each on it's respective unit cube
        # relevant_parents: torch.Tensor
        #   List of relevant parent indizes
        # parentweights: torch.Tensor
        #   Prior of each parent
        gmcount = relevant_parents.shape[0]
        gmdata = self.GMLevelTrainingData(self._dtype)
        gmdata.parents = relevant_parents.repeat(1, self._n_gaussians_per_node)\
            .view(self._n_gaussians_per_node, -1).transpose(-1, -2).reshape(1, 1, -1)
        position_templates = torch.tensor([
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 1]
        ], dtype=self._dtype, device='cuda')
        if self._n_gaussians_per_node <= 8:
            gmdata.positions = position_templates[0:self._n_gaussians_per_node].unsqueeze(0).unsqueeze(0)\
                .repeat(1, 1, gmcount, 1)
        else:
            gmdata.positions = position_templates.unsqueeze(0).unsqueeze(0). \
                repeat(1, 1, math.ceil(self._n_gaussians_per_node / 8), 1)
            gmdata.positions = gmdata.positions[:, :, 0:self._n_gaussians_per_node, :].repeat(1, 1, gmcount, 1)
        gmdata.covariances = 0.1 * torch.eye(3, dtype=self._dtype, device='cuda').unsqueeze(0).unsqueeze(0).unsqueeze(0). \
            repeat(1, 1, self._n_gaussians_per_node * gmcount, 1, 1)
        gmdata.priors = torch.zeros(1, 1, self._n_gaussians_per_node * gmcount, dtype=self._dtype, device='cuda')
        gmdata.priors[:, :, :] = 1 / self._n_gaussians_per_node

        gmdata.parentweights = parentweights

        return gmdata

    def _initialize_per_subgm(self, relevant_parents: torch.Tensor, parentweights: torch.Tensor,
                              parent_per_point: torch.Tensor, pcbatch_scaled: torch.Tensor):
        # Initializes new GMs, each individually according to the chosen method
        gmcount = relevant_parents.shape[0]
        gausscount = gmcount * self._n_gaussians_per_node
        gmdata = self.GMLevelTrainingData(self._dtype)
        gmdata.positions = torch.zeros(1, 1, gausscount, 3, dtype=self._dtype, device='cuda')
        gmdata.covariances = torch.zeros(1, 1, gausscount, 3, 3, dtype=self._dtype, device='cuda')
        gmdata.priors = torch.zeros(1, 1, gausscount, dtype=self._dtype, device='cuda')
        for i in relevant_parents:
            gidx_start = i * self._n_gaussians_per_node
            gidx_end = gidx_start + self._n_gaussians_per_node
            rel_point_mask: torch.Tensor = torch.eq(parent_per_point, i)
            relpoints = pcbatch_scaled[rel_point_mask]
            pcount = rel_point_mask.sum()
            if pcount < self._n_gaussians_per_node:
                gmdata.positions[0, 0, gidx_start:gidx_end] = 0.0
                gmdata.covariances[0, 0, gidx_start:gidx_end] = self._eps
                gmdata.priors[0, 0, gidx_start:gidx_end] = 0.0
                gmdata.positions[0, 0, gidx_start:gidx_start + pcount] = relpoints
                gmdata.covariances[0, 0, gidx_start:gidx_start + pcount] = 0.1 * torch.eye(3, dtype=self._dtype, device='cuda')
                gmdata.priors[0, 0, gidx_start:gidx_start + pcount] = 1 / pcount
            else:
                subgm = self._gmminitializer.initialize_by_method_name(self._initialization_method,
                                                                       relpoints.unsqueeze(0),
                                                                       self._n_gaussians_per_node)
                gmdata.positions[0, 0, gidx_start:gidx_end] = gm.positions(subgm)
                gmdata.covariances[0, 0, gidx_start:gidx_end] = gm.covariances(subgm)
                gmdata.priors[0, 0, gidx_start:gidx_end] = gm.weights(subgm)

        gmdata.parentweights = parentweights

        return gmdata

    @staticmethod
    def _extract_bbs(pcbatch: torch.Tensor, relevant_parents: torch.Tensor, parent_per_point: torch.Tensor) -> torch.Tensor:
        # This gets a pcbatch of size (1, n, 3) and point_weighting_factors of size (n, K) (K = #parents)
        # This returns the bounding boxes for each parent (K,2,3) (0=min,1=extend)
        # The bounding box is a arbitrary box rather than a regular cube.
        n_parents = relevant_parents.shape[0]
        result = torch.zeros(n_parents, 2, 3, dtype=pcbatch.dtype, device='cuda')
        for i in relevant_parents:
            rel_point_mask: torch.Tensor = torch.eq(parent_per_point, i)
            rel_points = pcbatch[rel_point_mask]
            if rel_points.shape[0] > 0:
                rel_bbmin = torch.min(rel_points, dim=0)[0]
                rel_bbmax = torch.max(rel_points, dim=0)[0]
                result[i, 0, :] = rel_bbmin
                result[i, 1, :] = rel_bbmax - rel_bbmin
                zeroextend = (result[i, 1, :] == 0)
                if zeroextend.any():
                    result[i, 0, zeroextend] -= 0.01
                    result[i, 1, zeroextend] = 0.02
            else:
                result[i, 0, :] = torch.tensor([0.0, 0.0, 0.0])
                result[i, 1, :] = torch.tensor([1.0, 1.0, 1.0])
        return result

    def _initialize_gms_on_bounding_box(self, bbs: torch.Tensor, relevant_parents: torch.Tensor,
                                        parentweights: torch.Tensor, finished_gaussians: torch.Tensor):
        # Initializes new GMs, each on the corners of its points respective bounding boxes
        # relevant_parents: torch.Tensor
        #   List of relevant parent indizes
        # parentweights: torch.Tensor
        #   Prior of each parent
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
        ], dtype=self._dtype, device='cuda')
        if self._n_gaussians_per_node <= 8:
            gmdata.positions = position_templates[0:self._n_gaussians_per_node].unsqueeze(0).unsqueeze(0)\
                .repeat(1, 1, gmcount, 1)
        else:
            gmdata.positions = position_templates.unsqueeze(0).unsqueeze(0). \
                repeat(1, 1, math.ceil(self._n_gaussians_per_node / 8), 1)
            gmdata.positions = gmdata.positions[:, :, 0:self._n_gaussians_per_node, :].repeat(1, 1, gmcount, 1)
        gmdata.positions[0, 0, :, :] *= bbs_rep[:, 1, :]
        gmdata.positions[0, 0, :, :] += bbs_rep[:, 0, :]
        gmdata.covariances = 0.1 * torch.eye(3, dtype=self._dtype, device='cuda').unsqueeze(0).unsqueeze(0).unsqueeze(0). \
            repeat(1, 1, self._n_gaussians_per_node * gmcount, 1, 1)
        gmdata.covariances[0, 0, :, :, :] *= bbs_rep[:, 1, :].unsqueeze(2) ** 2
        gmdata.priors = torch.zeros(1, 1, self._n_gaussians_per_node * gmcount, dtype=self._dtype, device='cuda')
        gmdata.priors[:, :, :] = 1 / self._n_gaussians_per_node
        gmdata.priors[:, :, finished_gaussians] = 0

        gmdata.parentweights = parentweights

        return gmdata

    def _initialize_on_eigen_vectors(self, relevant_parents: torch.Tensor, parentweights: torch.Tensor,
                                     parent_per_point: torch.Tensor, pcbatch_scaled: torch.Tensor):
        gmcount = relevant_parents.shape[0]
        gausscount = gmcount * self._n_gaussians_per_node
        gmdata = self.GMLevelTrainingData(self._dtype)
        gmdata.positions = torch.zeros(1, 1, gausscount, 3, dtype=self._dtype, device='cuda')
        gmdata.covariances = torch.zeros(1, 1, gausscount, 3, 3, dtype=self._dtype, device='cuda')
        gmdata.priors = torch.zeros(1, 1, gausscount, dtype=self._dtype, device='cuda')

        position_templates3d = torch.tensor([
            [-1, -1, -1],
            [1, 1, 1],
            [-1, 1, -1],
            [1, -1, 1],
            [-1, -1, 1],
            [1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1]
        ], dtype=self._dtype, device='cuda')
        position_templates2d = torch.tensor([
            [-1, -1, 0],
            [1, 1, 0],
            [-1, 1, 0],
            [1, -1, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0]
        ], dtype=self._dtype, device='cuda')
        # position_templates = torch.tensor([
        #     [-1, 0, 0],
        #     [1, 0, 0],
        #     [0, -1, 0],
        #     [0, 1, 0],
        #     [0, 0, -1],
        #     [0, 0, 1],
        #     [-1, -1, -1],
        #     [-1, 1, 1]
        # ], dtype=self._dtype, device='cuda')

        for i in relevant_parents:
            gidx_start = i * self._n_gaussians_per_node
            gidx_end = gidx_start + self._n_gaussians_per_node
            rel_point_mask: torch.Tensor = torch.eq(parent_per_point, i)
            relpoints = pcbatch_scaled[rel_point_mask]
            pcount = rel_point_mask.sum()
            if pcount < self._n_gaussians_per_node:
                gmdata.positions[0, 0, gidx_start:gidx_end] = 0.0
                gmdata.covariances[0, 0, gidx_start:gidx_end] = self._eps
                gmdata.priors[0, 0, gidx_start:gidx_end] = 0.0
                gmdata.positions[0, 0, gidx_start:gidx_start + pcount] = relpoints
                gmdata.covariances[0, 0, gidx_start:gidx_start + pcount] = 0.1 * torch.eye(3, dtype=self._dtype,
                                                                                           device='cuda')
                gmdata.priors[0, 0, gidx_start:gidx_start + pcount] = 1 / pcount
            else:
                meanpos = relpoints.mean(dim=0, keepdim=True)
                diffs = (relpoints - meanpos.expand(pcount, 3)).unsqueeze(2)
                meancov = (diffs * diffs.transpose(-1, -2)).mean(dim=0)
                meanweight = 1.0 / self._n_gaussians_per_node
                eigenvalues, eigenvectors = torch.symeig(meancov, True)
                eigenvalues_sorted, indices = torch.sort(eigenvalues[:], dim=0, descending=True)
                eigenvectors_sorted = 4 * eigenvalues_sorted.unsqueeze(0).repeat(3,1) * eigenvectors[:, indices]
                if self._n_gaussians_per_node <= 8:
                    if eigenvalues_sorted[2] > 1e-8:
                        gmdata.positions[0, 0, gidx_start:gidx_end] = position_templates3d[0:self._n_gaussians_per_node]
                    else:
                        gmdata.positions[0, 0, gidx_start:gidx_end] = position_templates2d[0:self._n_gaussians_per_node]
                else:
                    if eigenvalues_sorted[2] > 1e-8:
                        gmdata.positions = position_templates3d.repeat(math.ceil(self._n_gaussians_per_node / 8))[
                                           0:self._n_gaussians_per_node]
                    else:
                        gmdata.positions = position_templates2d.repeat(math.ceil(self._n_gaussians_per_node / 8))[
                                           0:self._n_gaussians_per_node]
                gmdata.positions[0, 0, gidx_start:gidx_end] = torch.matmul(eigenvectors_sorted, gmdata.positions[0, 0, gidx_start:gidx_end].transpose(-1, -2)).transpose(-1, -2) + meanpos.squeeze()
                gmdata.covariances[0, 0, gidx_start:gidx_end] = meancov + self._eps
                gmdata.priors[0, 0, gidx_start:gidx_end] = meanweight

        gmdata.parentweights = parentweights

        return gmdata

    def _expectation(self, points: torch.Tensor, gm_data, parent_per_point: torch.Tensor) -> torch.Tensor:
        # This performs the Expectation step of the EM Algorithm. This calculates the responsibilities.
        # So the probabilities, how likely each point belongs to each gaussian.
        # The calculations are performed numerically stable in Log-Space!
        # Parameters:
        #   points_rep: torch.Tensor of shape (1, 1, n_points, 1, 3)
        #       This is a view of the point cloud
        #   gm_data: GMLevelTrainingData
        #       The current GM-object
        #   parent_per_point: torch.Tensor of shape (n)
        #       The parent index for each point
        # Returns:
        #   responsibilities: torch.Tensor of shape (1, 1, n, p*g) where g is self._n_gaussians_per_node (results in
        #       the number of all gaussians on this level)
        #       responsibilities for point-gaussian-combinations where the point does not belong to a
        #       gaussian's parent will be 0. Also note, that there might be Sub-GMs without points assigned to them.
        #
        #   Note that this does not support executing only parts of the responsibilities at once for memory optimization
        #   (as in the M-Step). It would be possible to implement this though, similar as in the E-Step of EMGenerator

        batch_size = points.shape[0]
        n_sample_points = points.shape[2]
        points_rep = points.expand(1, 1, n_sample_points, self._n_gaussians_per_node, 3)
        all_gauss_count = gm_data.positions.shape[2]

        # mask_indizes is a list of indizes of a) points with their corresponding b) gauss (child) indizes
        mask_indizes = torch.zeros(n_sample_points, 2, dtype=torch.long, device='cuda')
        mask_indizes[:, 0] = torch.arange(0, n_sample_points, dtype=torch.long, device='cuda')
        mask_indizes[:, 1] = parent_per_point.view(-1)
        mask_indizes = mask_indizes.repeat(1, 1, self._n_gaussians_per_node)
        mask_indizes[:, :, 1::2] *= self._n_gaussians_per_node
        mask_indizes[:, :, 1::2] += torch.arange(0, self._n_gaussians_per_node, dtype=torch.long, device='cuda')
        mask_indizes = mask_indizes.view(n_sample_points * self._n_gaussians_per_node, 2)

        # GM-Positions, expanded for each PC point. shape: (1, 1, np, ng, 3)
        gmpositions_rep = gm_data.positions.unsqueeze(2).expand(1, 1, n_sample_points, all_gauss_count, 3)
        gmpositions_rep = gmpositions_rep[:, :, mask_indizes[:, 0], mask_indizes[:, 1]]\
            .view(1, 1, n_sample_points, self._n_gaussians_per_node, 3)
        # GM-Inverse Covariances, expanded for each PC point. shape: (1, 1, np, ng, 3, 3)
        gmicovs_rep = gm_data.calculate_inversed_covariances().unsqueeze(2)\
            .expand(batch_size, 1, n_sample_points, all_gauss_count, 3, 3)
        gmicovs_rep = gmicovs_rep[:, :, mask_indizes[:, 0], mask_indizes[:, 1], :, :]\
            .view(1, 1, n_sample_points, self._n_gaussians_per_node, 3, 3)
        # Tensor of {PC-point minus GM-position}-vectors. shape: (1, 1, np, ng, 3, 1)
        grelpos = (points_rep - gmpositions_rep).unsqueeze(5)
        # Tensor of 0.5 times the Mahalanobis distances of PC points to Gaussians. shape: (1, 1, np, ng)
        expvalues = 0.5 * \
            torch.matmul(grelpos.transpose(-2, -1), torch.matmul(gmicovs_rep, grelpos)).squeeze(5).squeeze(4)
        # Logarithmized GM-Priors, expanded for each PC point. shape: (1, 1, np, ng)
        gmpriors_log_rep = \
            torch.log(gm_data.calculate_amplitudes().unsqueeze(2).expand(1, 1, n_sample_points, all_gauss_count))
        gmpriors_log_rep = gmpriors_log_rep[:, :, mask_indizes[:, 0], mask_indizes[:, 1]]\
            .view(1, 1, n_sample_points, self._n_gaussians_per_node)

        # The logarithmized likelihoods of each point for each gaussian. shape: (1, 1, np, ng)
        likelihood_log = gmpriors_log_rep - expvalues

        # Logarithmized Likelihood for each point given the GM. shape: (1, 1, np, 1)
        llh_sum = torch.logsumexp(likelihood_log, dim=3, keepdim=True)

        # Local responsibilities: Responsibilities of points to their corresponding gaussians only
        responsibilities_local = torch.exp(likelihood_log - llh_sum)  # (1, 1, np, ngLocal)
        # Global responsibilities: Responsibilities of points to all gaussians
        responsibilities_global = torch.zeros(1, 1, n_sample_points, all_gauss_count, dtype=self._dtype, device='cuda')
        responsibilities_global[:, :, mask_indizes[:, 0], mask_indizes[:, 1]] = responsibilities_local.view(1, 1, -1)

        return responsibilities_global

    def _maximization(self, points: torch.Tensor, responsibilities: torch.Tensor, gm_data):
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

        n_sample_points = points.shape[2]
        all_gauss_count = responsibilities.shape[3]
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
            t_0 = torch.zeros(1, 1, actual_gauss_subbatch_size, dtype=self._dtype, device='cuda')
            t_1 = torch.zeros(1, 1, actual_gauss_subbatch_size, 3, dtype=self._dtype, device='cuda')
            t_2 = torch.zeros(1, 1, actual_gauss_subbatch_size, 3, 3, dtype=self._dtype, device='cuda')

            # Iterate over Point-Subbatches
            for i_start in range(0, n_sample_points, point_subbatch_size):
                i_end = i_start + point_subbatch_size
                relevant_responsibilities = responsibilities[:, :, i_start:i_end, j_start:j_end]
                actual_gauss_subbatch_size = relevant_responsibilities.shape[3]
                actual_point_subbatch_size = relevant_responsibilities.shape[2]

                points_rep = points[:, :, i_start:i_end]\
                    .expand(1, 1, actual_point_subbatch_size, actual_gauss_subbatch_size, 3)
                matrices_from_points = points_rep.unsqueeze(5) * points_rep.unsqueeze(5).transpose(-1, -2)

                # Fill T-Variables
                t_2 += (matrices_from_points
                        * relevant_responsibilities.unsqueeze(4).unsqueeze(5)).sum(dim=2)  # shape: (1, 1, J, 3, 3)
                t_0 += relevant_responsibilities.sum(dim=2)  # shape: (1, 1, J)
                t_1 += (points_rep * relevant_responsibilities.unsqueeze(4))\
                    .sum(dim=2)  # shape: (1, 1, J, 3)
                del matrices_from_points
                del points_rep

            # formulas taken from eckart paper
            gm_data.positions[:, :, j_start:j_end] = t_1 / t_0.unsqueeze(3)  # (1, 1, J, 3)
            gm_data.covariances[:, :, j_start:j_end] = t_2 / t_0.unsqueeze(3).unsqueeze(4) - \
                (gm_data.positions[:, :, j_start:j_end].unsqueeze(4) *
                    gm_data.positions[:, :, j_start:j_end].unsqueeze(4).transpose(-1, -2)) \
                + self._eps.expand_as(t_2)
            relevant_point_count = t_0.view(1, -1, self._n_gaussians_per_node).sum(dim=2) \
                .repeat(1, self._n_gaussians_per_node, 1).transpose(-1, -2).reshape(1, -1)
            gm_data.priors[:, :, j_start:j_end] = t_0 / relevant_point_count
            del t_0, t_1, t_2

        # Handling of invalid Gaussians! If all responsibilities of a Gaussian are zero, the previous code will
        # set the prior of it to zero and the covariances and positions to NaN
        # To avoid NaNs, we will then replace those invalid values with 0 (pos) and eps (cov).
        nans = torch.isnan(gm_data.priors) | (gm_data.priors == 0)
        gm_data.positions[nans] = torch.tensor([0.0, 0.0, 0.0], dtype=self._dtype, device='cuda')
        gm_data.covariances[nans] = self._eps[0, 0, 0, :, :]
        gm_data.priors[nans] = 0

    @staticmethod
    def _construct_full_gm(current_gm_upscaled_packed: torch.Tensor, finished_subgmms: list):
        for subgmm in finished_subgmms:
            current_gm_upscaled_packed = torch.cat((current_gm_upscaled_packed, subgmm), dim=2)
        current_gm_upscaled_packed = current_gm_upscaled_packed[:, :, gm.weights(current_gm_upscaled_packed)[0, 0] > 0]
        return current_gm_upscaled_packed

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

        def clone(self):
            cl = EckartGeneratorHP.GMLevelTrainingData(self.positions.dtype)
            cl.positions = self.positions.clone()
            cl.priors = self.priors.clone()
            cl.covariances = self.covariances.clone()
            cl.parents = self.parents.clone()
            cl.parentweights = self.parentweights.clone()
            return cl

        def calculate_inversed_covariances(self) -> torch.Tensor:
            return mat_tools.inverse(self.covariances).contiguous()

        def get_premultiplied_priors(self) -> torch.Tensor:
            # Returns the priors multiplied with all their parents priors
            return self.parentweights * self.priors

        def calculate_amplitudes(self) -> torch.Tensor:
            return self.priors / (self.covariances.det().sqrt() * 15.74960995)

        def approximate_whole_mixture(self, scaler: LevelScaler) -> torch.Tensor:
            # Scales the mixtures up and multiplies the priors with their parents priors
            # to generate a (more or less) valid Gaussian Mixture (with amplitudes).
            # It's not completely accurate, as all children of a Gaussian can have priors 0.
            # Usually these would be replaced with their parent. This is not happening.
            # It could therefore even be, that the weights do not sum to 0, so it's only an approximation.
            newpriors, newpositions, newcovariances = \
                scaler.unscale_gmm_wpc(self.priors, self.positions, self.covariances)
            newamplitudes = newpriors / (newcovariances.det().sqrt() * 15.74960995)
            return gm.pack_mixture(self.parentweights * newamplitudes, newpositions, newcovariances)

        def scale_up(self, scaler: LevelScaler):
            # Scales the Gaussian up given the LevelScaler
            self.priors, self.positions, self.covariances = \
                scaler.unscale_gmm_wpc(self.priors, self.positions, self.covariances)

        def pack_scaled_up_mixture(self, scaler:LevelScaler):
            npriors, npositions, ncovariances = \
                scaler.unscale_gmm_wpc(self.priors, self.positions, self.covariances)
            namplitudes = npriors / (ncovariances.det().sqrt() * 15.74960995)
            return gm.pack_mixture(self.parentweights * namplitudes, npositions, ncovariances)

        def __len__(self):
            # Returs the number of Gaussians in this level
            return self.priors.shape[2]
