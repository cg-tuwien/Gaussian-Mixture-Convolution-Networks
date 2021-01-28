import gmc.mixture as gm
import torch
from gmc import mat_tools


class EMTools:
    # This class capsules the core EM functionality, which is used by both
    # EMGenerator and GMMInitializer (which is why we move them into their own class)

    @staticmethod
    def expectation(points: torch.Tensor, gm_data, n_gaussians: int, running: torch.Tensor,
                    em_gaussians_subbatchsize: int = -1, em_points_subbatchsize: int = -1,
                    losses: torch.Tensor = None) -> \
            (torch.Tensor, torch.Tensor):
        # This performs the Expectation step of the EM Algorithm. This calculates 1) the responsibilities.
        # So the probabilities, how likely each point belongs to each gaussian and 2) the overall Log-Likelihood
        # of this GM given the point cloud.
        # The calculations are performed numerically stable in Log-Space!
        # Per default, all points and Gaussians are processed at once.
        # However, by setting em_gaussians_subbatchsize and em_points_subbatchsize,
        # this can be split into several processings to save memory.
        # Parameters:
        #   points: torch.Tensor of shape (batch_size, 1, n_points, 1, 3)
        #       This is a expansion of the (sampled) point cloud
        #   gm_data: TrainingData
        #       The current GM-object
        #   running: torch.Tensor of shape (batch_size), dtype=bool
        #       Gives information on which GMs are still running (true) or finished (false). Only relevant GMs will
        #       be considered. Responsibilities of GMs that are finished will be zero.
        #   em_gaussian_subbatchsize: int
        #       How many Gaussian Sub-Mixtures should be processed at once
        #       -1 means all Gaussians (default)
        #   em_points_subbatchsize: int
        #       How many points should be processed at once
        #       -1 means all Points (default)
        #   losses: torch.Tensor of shape (batch_size)
        #       Losses from last iteration. Will be returned for all the GMs that are not updated anymore
        #       Can also be None, then 1 will be returned for inactive GMs.
        # Returns:
        #   responsibilities: torch.Tensor of shape (batch_size, 1, n_points, n_gaussians)
        #   losses: torch.Tensor of shape (batch_size): Negative Log-Likelihood for each GM

        batch_size = points.shape[0]
        running_batch_size = running.sum()
        n_sample_points = points.shape[2]
        dtype = points.dtype

        has_noise = gm_data.has_noise_cluster()

        # This uses the fact that
        # log(a * exp(-0.5 * M(x))) = log(a) + log(exp(-0.5 * M(x))) = log(a) - 0.5 * M(x)

        gauss_subbatch_size = em_gaussians_subbatchsize
        if gauss_subbatch_size < 1:
            gauss_subbatch_size = n_gaussians
        point_subbatch_size = em_points_subbatchsize
        if point_subbatch_size < 1:
            point_subbatch_size = n_sample_points

        likelihood_log = \
            torch.zeros(running_batch_size, 1, n_sample_points, n_gaussians + has_noise, dtype=dtype, device='cuda')
        gmpos = gm_data.get_positions()
        gmicov = gm_data.get_inversed_covariances()
        gmloga = gm_data.get_logarithmized_amplitudes()

        for j_start in range(0, n_gaussians, gauss_subbatch_size):
            j_end = j_start + gauss_subbatch_size
            actual_gauss_subbatch_size = min(n_gaussians, j_end) - j_start
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
        if has_noise:
            for i_start in range(0, n_sample_points, point_subbatch_size):
                i_end = i_start + point_subbatch_size
                actual_point_subbatch_size = min(n_sample_points, i_end) - i_start
                likelihood_log[:, :, i_start:i_end, -1] = gm_data.get_noise_loglikelihood()\
                    .unsqueeze(1).unsqueeze(2).expand(running_batch_size, 1, actual_point_subbatch_size)

        # Logarithmized Likelihood for each point given the GM. shape: (bs, 1, np, ng)
        llh_sum = torch.logsumexp(likelihood_log, dim=3, keepdim=True)
        # Logarithmized Mean Likelihood for all points. shape: (bs)
        if losses is None:
            losses = torch.zeros(batch_size, dtype=dtype, device='cuda')
        losses[running] = -llh_sum.mean(dim=2).view(running_batch_size)
        # Calculating responsibilities
        responsibilities = torch.zeros(batch_size, 1, n_sample_points, n_gaussians + has_noise, dtype=dtype,
                                       device='cuda')
        responsibilities[running] = torch.exp(likelihood_log - llh_sum)

        # Calculating responsibilities and returning them and the mean loglikelihoods
        return responsibilities, losses

    @staticmethod
    def maximization(points_rep: torch.Tensor, responsibilities: torch.Tensor, gm_data, running: torch.Tensor,
                     eps: torch.Tensor, em_gaussians_subbatchsize: int = -1, em_points_subbatchsize: int = -1):
        # This performs the Maximization step of the EM Algorithm.
        # Updates the GM-Model given the responsibilities which resulted from the E-Step.
        # Per default, all points and Gaussians are processed at once.
        # However, by setting em_gaussians_subbatchsize and em_points_subbatchsize,
        # this can be split into several processings to save memory.
        # Parameters:
        #   points_rep: torch.Tensor of shape (batch_size, 1, n_points, n_gaussians, 3)
        #       This is a expansion of the (sampled) point cloud, repeated n_gaussian times along dimension 4
        #   responsibilities: torch.Tensor of shape (batch_size, 1, n_points, n_gaussians)
        #       This is the result of the E-step.
        #   gm_data: EMTools.TrainingData
        #       The current GM-object (will be changed)
        #   running: torch.Tensor of shape (batch_size), dtype=bool
        #       Gives information on which GMs are still running (true) or finished (false).
        #   eps: torch.Tensor of shape (batch_size, 1, 1, 3, 3)
        #       Small-valued matrix to add to all the calculated covariance matrices to avoid numerical problems
        #   em_gaussian_subbatchsize: int
        #       How many Gaussian Sub-Mixtures should be processed at once
        #       -1 means all Gaussians (default)
        #   em_points_subbatchsize: int
        #       How many points should be processed at once
        #       -1 means all Points (default)

        n_sample_points = points_rep.shape[2]
        n_gaussians = points_rep.shape[3]
        dtype = points_rep.dtype
        n_running = running.sum()
        gauss_subbatch_size = em_gaussians_subbatchsize
        if gauss_subbatch_size < 1:
            gauss_subbatch_size = n_gaussians
        point_subbatch_size = em_points_subbatchsize
        if point_subbatch_size < 1:
            point_subbatch_size = n_sample_points

        new_positions = gm_data.get_positions()[running].clone()
        new_covariances = gm_data.get_covariances()[running].clone()
        new_priors = gm_data.get_priors()[running].clone()
        new_noise_weight = gm_data.get_noise_weight()[running]

        # Iterate over Gauss-Subbatches
        for j_start in range(0, n_gaussians, gauss_subbatch_size):
            j_end = j_start + gauss_subbatch_size
            actual_gauss_subbatch_size = min(n_gaussians, j_end) - j_start
            # Initialize T-Variables for these Gaussians, will be filled in the upcoming loop
            # Positions/Covariances/Priors are calculated from these (see Eckart-Paper)
            t_0 = torch.zeros(n_running, 1, actual_gauss_subbatch_size, dtype=dtype, device='cuda')
            t_1 = torch.zeros(n_running, 1, actual_gauss_subbatch_size, 3, dtype=dtype, device='cuda')

            # Iterate over Point-Subbatches
            for i_start in range(0, n_sample_points, point_subbatch_size):
                i_end = i_start + point_subbatch_size
                relevant_responsibilities = responsibilities[running, :, i_start:i_end, j_start:j_end]
                # actual_point_subbatch_size = relevant_responsibilities.shape[2]
                relevant_points = points_rep[running, :, i_start:i_end, j_start:j_end, :]
                # Fill T-Variables      # t_2 shape: (1, 1, J, 3, 3)
                t_0 += relevant_responsibilities.sum(dim=2)  # shape: (1, 1, J)
                t_1 += (relevant_points * relevant_responsibilities.unsqueeze(4)) \
                    .sum(dim=2)  # shape: (1, 1, J, 3)
                del relevant_points

            new_priors[:, :, j_start:j_end] = t_0 / n_sample_points
            new_positions[:, :, j_start:j_end] = t_1 / t_0.unsqueeze(3)  # (bs, 1, ng, 3)

            del t_1
            t_2 = torch.zeros(n_running, 1, actual_gauss_subbatch_size, 3, 3, dtype=dtype, device='cuda')

            for i_start in range(0, n_sample_points, point_subbatch_size):
                i_end = i_start + point_subbatch_size
                relevant_responsibilities = responsibilities[running, :, i_start:i_end, j_start:j_end]
                actual_point_subbatch_size = relevant_responsibilities.shape[2]
                relevant_relative_points = points_rep[running, :, i_start:i_end, j_start:j_end, :] \
                    - new_positions[:, :, j_start:j_end].unsqueeze(2)\
                    .expand(n_running, 1, actual_point_subbatch_size, actual_gauss_subbatch_size, 3)
                matrices_from_points = relevant_relative_points.unsqueeze(5) * relevant_relative_points.unsqueeze(5)\
                    .transpose(-1, -2)
                t_2 += (matrices_from_points * relevant_responsibilities.unsqueeze(4).unsqueeze(5)).sum(dim=2)
                del matrices_from_points, relevant_relative_points

            new_covariances[:, :, j_start:j_end] = t_2 / t_0.unsqueeze(3).unsqueeze(4) + eps[running]
            del t_0, t_2

        if gm_data.has_noise_cluster():
            new_noise_weight[running] = responsibilities[running, 0, :, -1].sum(dim=1) / n_sample_points

        # Handling of invalid Gaussians! If all responsibilities of a Gaussian are zero, the previous code will
        # set the prior of it to zero and the covariances and positions to NaN
        # To avoid NaNs, we will then replace those invalid values with 0 (pos) and eps (cov).
        if (new_priors == 0).sum() > 0:
            print("detected ", (new_priors == 0).sum().item(), "0-priors!")
        new_positions[new_priors == 0] = torch.tensor([0.0, 0.0, 0.0], dtype=dtype, device='cuda')
        new_covariances[new_priors == 0] = eps[0, 0, 0, :, :]

        # Update GMData
        gm_data.set_positions(new_positions, running)
        gm_data.set_covariances(new_covariances, running)
        gm_data.set_priors(new_priors, running)
        gm_data.set_noise_weight(new_noise_weight, running)

    @staticmethod
    def find_valid_matrices(covariances: torch.Tensor, invcovs: torch.Tensor, strong: bool = False) -> torch.Tensor:
        # Returns a boolean tensor describing which of the given covariances are valid positice definite matrices.
        # Parameters:
        #   covariances: torch.Tensor of size (bs, 1, ng, 3, 3)
        #       Tensor of covariances.
        #   invcovs: torch.Tensor of size (bs, 1, ng, 3, 3)
        #       Inverses of covariances. These need to be checked too as numerical instabilities might result in
        #       invalid inverses.
        #   strong: bool = False
        #       Usually, all matrices are checked at once. If strong is true, each matrix is checked individually.
        #       This shouldn't make a difference in most cases. There are only very rare cases where the results
        #       are different. However, this option takes much longer, so it is not very usable in practice.
        # returns a bool-tensor of size (bs, 1, ng)
        relcovs = ~(torch.isnan(covariances.det().sqrt()) | covariances[:, :, :, 0:2, 0:2].det().lt(0)
                    | covariances[:, :, :, 0, 0].lt(0) | invcovs.det().lt(0) |
                    invcovs[:, :, :, 0:2, 0:2].det().lt(0) | invcovs[:, :, :, 0, 0].lt(0))
        # # More reliable way to check!
        if strong:
            asd = relcovs.clone()
            for i in range(covariances.shape[0]):
                for j in range(covariances.shape[2]):
                    relcovs[i, :, j] &= ~covariances[i, :, j].det().sqrt().isnan().any()
                    relcovs[i, :, j] &= ~covariances[i, :, j, 0:2, 0:2].det().lt(0).any()
                    relcovs[i, :, j] &= ~covariances[i, :, j, 0, 0].lt(0).any()
                    relcovs[i, :, j] &= ~invcovs[i, :, j].det().sqrt().isnan().any()
                    relcovs[i, :, j] &= ~invcovs[i, :, j, 0:2, 0:2].det().lt(0).any()
                    relcovs[i, :, j] &= ~invcovs[i, :, j, 0, 0].lt(0).any()
            if not asd.eq(relcovs).all():
                print("strong ditching was active!")
        return relcovs

    @staticmethod
    def replace_invalid_matrices(covariances: torch.Tensor, invcovs: torch.Tensor, defaultcov: torch.Tensor,
                                 defaulticov: torch.Tensor = None, strong: bool = False):
        # Replaces all given non-positive-definite matrices by a default matrix.
        # If a matrix's inverse is not positive-definite both the inverse and the original matrix itself is replaced
        # and vice versa (as long as defaulticov is given).
        # Parameters:
        #   covariances: torch.Tensor of size (bs, 1, ng, 3, 3)
        #       Covariance-Matrices to check and potentially replace
        #   invcovs: torch.Tensor of size (bs,1 , ng, 3, 3)
        #       Inverses of covariances
        #   defaultcov: torch.Tensor of size (bs, 1, 1, 3, 3)
        #       Default covariance matrix to replace invalid ones with (per batch)
        #   defaulticov: torch.Tensor of size (bs, 1, 1, 3, 3) = None
        #       Default inverse covariance matrix to replace invalid ones with (per batch)
        #       If this is None, the inverse covariances are not replaced.
        #   strong: bool = False
        #       See findValidMatrices for info.
        relcovs = EMTools.find_valid_matrices(covariances, invcovs, strong)
        defaultcov_expand = defaultcov.expand(covariances.size())
        covariances[~relcovs] = defaultcov_expand[~relcovs]
        if defaulticov is not None:
            defaulticov_expand = defaulticov.expand(invcovs.size())
            invcovs[~relcovs] = defaulticov_expand[~relcovs]

    class TrainingData:
        # Helper class. Capsules all relevant training data of the current GM batch.
        # positions, covariances and priors are stored as-is and can be set.
        # inversed covariances are calcualted whenever covariances are set.
        # amplitudes are calculated from priors (or vice versa).
        # Note that priors or amplitudes should always be set after the covariances are set,
        # otherwise the conversion is not correct anymore.

        def __init__(self, batch_size, n_gaussians, dtype, epsilons):
            self._positions = torch.zeros(batch_size, 1, n_gaussians, 3, dtype=dtype, device='cuda')
            self._logamplitudes = torch.zeros(batch_size, 1, n_gaussians, dtype=dtype, device='cuda')
            self._priors = torch.zeros(batch_size, 1, n_gaussians, dtype=dtype, device='cuda')
            self._covariances = torch.eye(3, 3, dtype=dtype, device='cuda').view(1, 1, 1, 3, 3)\
                .repeat(batch_size, 1, n_gaussians, 1, 1) * epsilons
            self._inversed_covariances = mat_tools.inverse(self._covariances).contiguous()
            self._noise_weight = torch.zeros(batch_size, dtype=dtype, device='cuda')
            self._noise_val = torch.zeros(batch_size, dtype=dtype, device='cuda')

        def set_positions(self, positions, running):
            # running indicates which batch entries should be replaced
            self._positions[running] = positions

        def set_covariances(self, covariances, running):
            # running indicates which batch entries should be replaced
            # If changing the covariance would lead them to loose their positive definiteness, they will not be changed
            # (Det-Ditching)

            invcovs = mat_tools.inverse(covariances).contiguous()
            relcovs = EMTools.find_valid_matrices(covariances, invcovs)
            # if (~relcovs).sum() != 0:
            #     print("ditching ", (~relcovs).sum().item(), " items")
            runningcovs = self._covariances[running]
            runningcovs[relcovs] = covariances[relcovs]
            self._covariances[running] = runningcovs
            runningicovs = self._inversed_covariances[running]
            runningicovs[relcovs] = invcovs[relcovs]
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

        def has_noise_cluster(self) -> bool:
            return self._noise_weight.gt(0).any().item()

        def get_noise_weight(self):
            return self._noise_weight

        def get_noise_value(self):
            return self._noise_val

        def get_noise_loglikelihood(self):
            return torch.log(self._noise_weight) + torch.log(self._noise_val)

        def set_noise(self, noise_weight, noise_val):
            self._noise_weight = noise_weight
            self._noise_val = noise_val

        def set_noise_weight(self, noise_weight, running):
            self._noise_weight[running] = noise_weight

        def set_noise_val(self, noise_val):
            self._noise_val = noise_val

        def pack_mixture(self):
            return gm.pack_mixture(torch.exp(self._logamplitudes), self._positions, self._covariances)

        def pack_mixture_model(self):
            return gm.pack_mixture(self._priors, self._positions, self._covariances)
