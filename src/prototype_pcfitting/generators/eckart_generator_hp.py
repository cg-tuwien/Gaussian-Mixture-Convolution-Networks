from prototype_pcfitting import GMMGenerator, GMLogger, data_loading, Scaler, ScalingMethod
from prototype_pcfitting import TerminationCriterion, MaxIterationTerminationCriterion
from gmc.cpp.extensions.furthest_point_sampling import furthest_point_sampling
from prototype_pcfitting.error_functions import LikelihoodLoss
from .level_scaler2 import LevelScaler2
import torch
import gmc.mixture as gm
import math, gc


class EckartGeneratorHP(GMMGenerator):
    # GMM Generator using Expectation Sparsification with hard partitioning by Eckart

    def __init__(self,
                 n_gaussians_per_node: int,
                 n_levels: int,
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
        self._termination_criterion = termination_criterion
        self._m_step_gaussians_subbatchsize = m_step_gaussians_subbatchsize
        self._m_step_points_subbatchsize = m_step_points_subbatchsize
        self._dtype = dtype

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
        # It returns two gaussian mixtures of sizes
        # [m,1,g,10], the first being a mixture with amplitudes as weights
        # the second a mixture where the weights describe the priors.
        # Training parameters have to be set in the other methods
        # of the class
        assert (gmbatch is None), "EckartGenerator cannot improve existing GMMs"

        batch_size = pcbatch.shape[0]

        assert (batch_size is 1), "EckartGenerator currently does not support batchsizes > 1"
        point_count = pcbatch.shape[1]
        pcbatch = pcbatch.to(self._dtype).cuda()

        # eps is used to avoid singularities and as default covariance for invalid gaussians
        self._eps = (torch.eye(3, 3, dtype=self._dtype) * 1e-6).view(1, 1, 1, 3, 3).cuda()
        # lossfunction for calculating the accurate loss for
        lossfunction = LikelihoodLoss()

        parent_per_point = torch.zeros(1, point_count).to(torch.long).cuda()
        parent_per_point[:, :] = 0

        hierarchy = []
        parentweights = torch.ones(1, 1, self._n_gaussians_per_node).cuda()

        absiteration = 0
        for l in range(self._n_levels):
            print("Level: ", l)
            parentcount_for_level = self._n_gaussians_per_node ** l
            gausscount_for_level = self._n_gaussians_per_node ** (l+1)
            relevant_parents = torch.arange(0, parentcount_for_level).cuda() # (gausscount)

            scaler = LevelScaler2()
            scaler.set_pointcloud(pcbatch, parent_per_point, parentcount_for_level, self._n_gaussians_per_node)
            points_scaled = scaler.scale_down_pc(pcbatch)       # (1, np, 3)

            gm_data = self._initialize_gms_on_unit_cube(relevant_parents, parentweights)

            self._termination_criterion.reset()

            iteration = 0
            while True:
                iteration += 1
                absiteration += 1

                points = points_scaled.unsqueeze(1).unsqueeze(3) # (1, 1, np, 1, 3)

                responsibilities, losses = self._expectation(points, gm_data, parent_per_point, relevant_parents)
                # point_count_per_parent = responsibilities.sum(dim=2).view(-1, self._n_gaussians_per_node).sum(dim=1)
                losses = scaler.scale_up_losses(losses)
                loss = -losses[~torch.isnan(losses)].mean().view(1)
                del losses

                if self._logger:
                    mixture = gm_data.approximate_whole_mixture(scaler)
                    self._logger.log(absiteration - 1, loss, mixture)
                    del mixture
                    gc.collect()

                if not self._termination_criterion.may_continue(iteration - 1, loss.view(-1)).item():
                    # update parent_per_point for next level from responsibilities
                    new_parent_indices = responsibilities.argmax(dim=3)
                    parent_per_point = new_parent_indices.view(1, -1)
                    break

                self._maximization(points, responsibilities, gm_data)

            gm_data.scale_up(scaler)
            # add gm to hierarchy
            hierarchy.append(gm_data)
            parentweights = gm_data.get_premultiplied_priors().repeat(1, 1, self._n_gaussians_per_node, 1).transpose(-1, -2).reshape(1, 1, -1)

        res_gm, res_gmm = self._construct_gm_from_hierarchy(hierarchy)
        res_gm = res_gm.float()
        res_gmm = res_gmm.float()

        loss = LikelihoodLoss()
        print("Final Loss: ", loss.calculate_score_packed(pcbatch, res_gm))

        return res_gm, res_gmm

    def _construct_gm_from_hierarchy(self, hierarchy) -> (torch.Tensor, torch.Tensor):
        priors, covariances, positions = self._expand_subgm_from_level(hierarchy, -1, 0)
        resgmm = gm.pack_mixture(priors, positions, covariances)
        resgm = gm.convert_priors_to_amplitudes(resgmm)
        return resgm, resgmm

    def _expand_subgm_from_level(self, hierarchy, level, index) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        if level == len(hierarchy) - 1:
            gmdata = hierarchy[-1]
            return gmdata.priors[:, :, index:index+1], gmdata.covariances[:, :, index:index+1], \
                   gmdata.positions[:, :, index:index+1]
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
                endindex = currentindex+num
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
                    return empty, empty.unsqueeze(3).unsqueeze(4).repeat(1, 1, 0, 3, 3), empty.unsqueeze(3).repeat(1, 1, 0, 3)
                return gmdata_upper.priors[:, :, index:index + 1], \
                       gmdata_upper.covariances[:, :, index:index + 1], \
                       gmdata_upper.positions[:, :, index:index + 1]


    def _initialize_gms_on_unit_cube(self, relevant_parents, parentweights):
        gmcount = relevant_parents.shape[0]
        gmdata = self.GMLevelTrainingData()
        gmdata.parents = relevant_parents.repeat(1, self._n_gaussians_per_node).view(self._n_gaussians_per_node, -1).transpose(-1, -2).reshape(1, 1, -1)
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
            gmdata.positions = position_templates[0:self._n_gaussians_per_node].unsqueeze(0).unsqueeze(0).repeat(1, 1,
                                                                                                            gmcount, 1)
        else:
            gmdata.positions = position_templates.unsqueeze(0).unsqueeze(0). \
                repeat(1, 1, math.ceil(self._n_gaussians_per_node / 8), 1)
            gmdata.positions = gmdata.positions[:, :, 0:self._n_gaussians_per_node, :].repeat(1, 1, gmcount, 1)
        gmdata.covariances = 0.1 * torch.eye(3).to(self._dtype).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(0). \
            repeat(1, 1, self._n_gaussians_per_node * gmcount, 1, 1)
        gmdata.priors = torch.zeros(1, 1, self._n_gaussians_per_node * gmcount).to(self._dtype).cuda()
        gmdata.priors[:, :, :] = 1 / self._n_gaussians_per_node

        gmdata.parentweights = parentweights

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
        points_rep = points.expand(1, 1, n_sample_points, self._n_gaussians_per_node, 3)
        all_gauss_count = gm_data.positions.shape[2]
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

        # GM-Positions, expanded for each PC point. shape: (bs, 1, np, ng, 3)
        gmpositions_rep = gm_data.positions.unsqueeze(2).expand(1, 1, n_sample_points, all_gauss_count, 3)
        gmpositions_rep = gmpositions_rep[:, :, mask_indizes[:, 0], mask_indizes[:, 1]].view(1, 1, n_sample_points, self._n_gaussians_per_node, 3)
        # GM-Inverse Covariances, expanded for each PC point. shape: (bs, 1, np, ng, 3, 3)
        gmicovs_rep = gm_data.calculate_inversed_covariances().unsqueeze(2).expand(batch_size, 1, n_sample_points, all_gauss_count, 3, 3)
        gmicovs_rep = gmicovs_rep[:, :, mask_indizes[:, 0], mask_indizes[:, 1], :, :].view(1, 1,
                                                                                           n_sample_points,
                                                                                           self._n_gaussians_per_node,
                                                                                           3, 3)
        # Tensor of {PC-point minus GM-position}-vectors. shape: (bs, 1, np, ng, 3, 1)
        grelpos = (points_rep - gmpositions_rep).unsqueeze(5)
        # Tensor of 0.5 times the Mahalanobis distances of PC points to Gaussians. shape: (bs, 1, np, ng)
        expvalues = 0.5 * \
                    torch.matmul(grelpos.transpose(-2, -1), torch.matmul(gmicovs_rep, grelpos)).squeeze(5).squeeze(4)
        # Logarithmized GM-Priors, expanded for each PC point. shape: (bs, 1, np, ng)
        gmpriors_log_rep = \
            torch.log(gm_data.calculate_amplitudes().unsqueeze(2).expand(1, 1, n_sample_points, all_gauss_count))
        gmpriors_log_rep = gmpriors_log_rep[:, :, mask_indizes[:, 0], mask_indizes[:, 1]].view(1, 1, n_sample_points, self._n_gaussians_per_node)
        parentpriors_log_rep = torch.log(gm_data.parentweights.unsqueeze(2).expand(1, 1, n_sample_points, all_gauss_count))
        parentpriors_log_rep = parentpriors_log_rep[:, :, mask_indizes[:, 0], mask_indizes[:, 1]].view(batch_size, 1, n_sample_points, self._n_gaussians_per_node)

        # The logarithmized likelihoods of each point for each gaussian. shape: (bs, 1, np, ng)
        likelihood_log = gmpriors_log_rep - expvalues
        global_likelihood_log = parentpriors_log_rep + likelihood_log

        # Logarithmized Likelihood for each point given the GM. shape: (bs, 1, np, 1)
        llh_sum = torch.logsumexp(likelihood_log, dim=3, keepdim=True)
        global_llh_sum = torch.logsumexp(global_likelihood_log, dim=3, keepdim=True)

        # Calculate Loss per GM (per Parent)
        # das passt nicht, denn wir haben den falschen gausscount. wir wollens pro parent
        indicator = parent_per_point.unsqueeze(2).repeat(1, 1, parent_count)    # (1, np, nP)
        indicator = (indicator == (torch.arange(0, parent_count).cuda().view(1, 1, -1).repeat(1, n_sample_points, 1)))
        losses = (indicator * global_llh_sum.squeeze(1).repeat(1, 1, parent_count)).sum(dim=1) / indicator.sum(dim=1) # (1, ng)

        # assert not torch.isnan(losses).any()
        # losses[torch.isnan(losses)] = 0 # temporary fix: as there are parents without gaussians there will be no loss defined for some
        # let them be nan and just ignore them for total loss calculation

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
        gauss_subbatch_size = self._n_gaussians_per_node * self._m_step_gaussians_subbatchsize
        if gauss_subbatch_size < 1:
            gauss_subbatch_size = all_gauss_count
        point_subbatch_size = self._m_step_points_subbatchsize
        if point_subbatch_size < 1:
            point_subbatch_size = n_sample_points
        for j_start in range(0, all_gauss_count, gauss_subbatch_size):
            j_end = j_start + gauss_subbatch_size
            actual_gauss_subbatch_size = min(all_gauss_count, j_end) - j_start
            T_0 = torch.zeros(1, 1, actual_gauss_subbatch_size, dtype=self._dtype).cuda()
            T_1 = torch.zeros(1, 1, actual_gauss_subbatch_size, 3, dtype=self._dtype).cuda()
            T_2 = torch.zeros(1, 1, actual_gauss_subbatch_size, 3, 3, dtype=self._dtype).cuda()

            for i_start in range(0, n_sample_points, point_subbatch_size):
                i_end = i_start + point_subbatch_size
                relevant_responsibilities = responsibilities[:,:,i_start:i_end,j_start:j_end]
                actual_gauss_subbatch_size = relevant_responsibilities.shape[3]
                actual_point_subbatch_size = relevant_responsibilities.shape[2]

                points_rep = points[:, :, i_start:i_end].expand(1, 1, actual_point_subbatch_size, actual_gauss_subbatch_size, 3)
                matrices_from_points = points_rep.unsqueeze(5) * points_rep.unsqueeze(5).transpose(-1, -2)

                T_2 += (matrices_from_points[:,:,:,0:actual_gauss_subbatch_size] * relevant_responsibilities.unsqueeze(4).unsqueeze(5)).sum(dim = 2) # shape: (1, 1, J, 3, 3)
                T_0 += relevant_responsibilities.sum(dim=2) # shape: (1, 1, J)
                T_1 += (points_rep[:,:,:,0:actual_gauss_subbatch_size] * relevant_responsibilities.unsqueeze(4)).sum(dim=2) # shape: (1, 1, J, 3)
                del matrices_from_points

            gm_data.positions[:,:,j_start:j_end] = T_1 / T_0.unsqueeze(3)  # (1, 1, J, 3)
            gm_data.covariances[:,:,j_start:j_end] = T_2 / T_0.unsqueeze(3).unsqueeze(4) - \
                                   (gm_data.positions[:,:,j_start:j_end].unsqueeze(4) * gm_data.positions[:,:,j_start:j_end].unsqueeze(4).transpose(-1, -2)) \
                                   + self._eps.expand_as(T_2)
            relevant_point_count = T_0.view(1, -1, self._n_gaussians_per_node).sum(dim=2) \
                .repeat(1, self._n_gaussians_per_node, 1).transpose(-1, -2).reshape(1, -1)
            gm_data.priors[:,:,j_start:j_end] = T_0 / relevant_point_count
            del T_0, T_1, T_2

        # Handling of invalid Gaussians! If all responsibilities of a Gaussian are zero, the previous code will
        # set the prior of it to zero and the covariances and positions to NaN
        # To avoid NaNs, we will then replace those invalid values with 0 (pos) and eps (cov).
        nans = torch.isnan(gm_data.priors) | (gm_data.priors == 0)
        gm_data.positions[nans] = torch.tensor([0.0, 0.0, 0.0], dtype=self._dtype).cuda()
        gm_data.covariances[nans] = self._eps[0, 0, 0, :, :]
        gm_data.priors[nans] = 0

    class GMLevelTrainingData:
        # Helper class. Capsules all relevant training data of the current GM batch on the given level.
        # positions, covariances and priors are stored as-is and can be set.
        # inversed covariances and amplitudes can be calculated from these.
        # Additionally, for each Gaussian, its parent Gauss index, and the product of all parents priors
        # are stored.
        # Note that this is not one GM, but a whole bunch of GMs managed in the same structure

        def __init__(self):
            self.positions : torch.Tensor = torch.tensor([])
            self.priors : torch.Tensor = torch.tensor([])
            self.covariances : torch.Tensor = torch.tensor([])
            self.parents = torch.tensor([])  # (1, 1, g) Indizes of parent Gaussians on parent Level
            self.parentweights = torch.tensor([]) # (1, 1, g) Combined prior-weights of all parents

        def calculate_inversed_covariances(self) -> torch.Tensor:
            return self.covariances.inverse().contiguous()

        def get_premultiplied_priors(self) -> torch.Tensor:
            # Returns the priors multiplied with all their parents priors
            return self.parentweights * self.priors

        def calculate_amplitudes(self) -> torch.Tensor:
            return self.priors / (self.covariances.det().sqrt() * 15.74960995)

        def approximate_whole_mixture(self, scaler: LevelScaler2) -> torch.Tensor:
            # Scales the mixtures up and multiplies the priors with their parents priors
            # to generate a (more or less) valid Gaussian Mixture (with amplitudes).
            # It's not completely accurate, as all children of a Gaussian can have priors 0.
            # Usually these would be replaced with their parent. This is not happening.
            # It could therefore even be, that the weights do not sum to 0, so it's only an approximation.
            newpriors, newpositions, newcovariances = \
                scaler.scale_up_gmm_wpc(self.priors, self.positions, self.covariances)
            newamplitudes = newpriors / (newcovariances.det().sqrt() * 15.74960995)
            return gm.pack_mixture(self.parentweights * newamplitudes, newpositions, newcovariances)

        def scale_up(self, scaler: LevelScaler2):
            # Scales the Gaussian up given the LevelScaler
            self.priors, self.positions, self.covariances = \
                scaler.scale_up_gmm_wpc(self.priors, self.positions, self.covariances)

        def __len__(self):
            # Returs the number of Gaussians in this level
            return self.priors.shape[2]
