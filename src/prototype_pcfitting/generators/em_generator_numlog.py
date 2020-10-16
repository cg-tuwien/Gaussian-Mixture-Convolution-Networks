from prototype_pcfitting import GMMGenerator, GMLogger, data_loading
from prototype_pcfitting import TerminationCriterion, MaxIterationTerminationCriterion
from prototype_pcfitting.error_functions import LikelihoodLoss
import torch
import gmc.mixture as gm
import numpy


class EMGeneratorNumLog(GMMGenerator):
    # GMM Generator using simple Expectation Maximization

    _device = torch.device('cuda')

    def __init__(self,
                 n_gaussians: int,
                 n_sample_points: int,
                 termination_criterion: TerminationCriterion = MaxIterationTerminationCriterion(500),
                 min_det: float = 1e-10):
        self._n_gaussians = n_gaussians
        self._n_sample_points = n_sample_points
        self._termination_criterion = termination_criterion
        self._min_det = min_det
        self._logger = None
        self._eps = None

    def set_logging(self, logger: GMLogger = None):
        self._logger = logger

    def generate(self, pcbatch: torch.Tensor, gmbatch: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        self._termination_criterion.reset()

        batch_size = pcbatch.shape[0]
        point_count = pcbatch.shape[1]
        n_sample_points = min(point_count, self._n_sample_points)
        pcbatch = pcbatch.to(self._device).double()  # dimension: (bs, np, 3)

        assert(point_count > self._n_gaussians)

        lossgen = LikelihoodLoss()

        # Initialize mixture (Important: Assumes intervall [0,1])
        if gmbatch is None:
            gmbatch = self.initialize(pcbatch)

        gm_data = self.TrainingData()
        gm_data.set_positions(gm.positions(gmbatch))
        gm_data.set_covariances(gm.covariances(gmbatch))
        gm_data.set_amplitudes(gm.weights(gmbatch))

        sample_points = data_loading.sample(pcbatch, n_sample_points)
        losses = lossgen.calculate_score(sample_points, gm_data.get_positions(), gm_data.get_covariances(),
                                          gm_data.get_inversed_covariances(), gm_data.get_amplitudes())
        loss = losses.sum()
        iteration = 0

        if self._logger:
            self._logger.log(iteration, losses, gm_data.pack_mixture())

        self._eps = (torch.eye(3, 3, dtype=torch.double) * 1e-6).view(1,1,1,3,3).expand(batch_size, 1, self._n_gaussians, 3, 3).cuda()

        while self._termination_criterion.may_continue(iteration, loss.item()):
            iteration += 1

            if n_sample_points < point_count:
                sample_points = data_loading.sample(pcbatch, n_sample_points)
            else:
                sample_points = pcbatch
            points_rep = sample_points.unsqueeze(1).unsqueeze(3).expand(batch_size, 1, n_sample_points, self._n_gaussians, 3)

            # Expectation
            responsibilities, llh = self.expectation(points_rep, gm_data)

            # Maximization
            self.maximization(points_rep, responsibilities, gm_data)

            # Calculate Loss on whole data set
            losses = lossgen.calculate_score(sample_points, gm_data.get_positions(), gm_data.get_covariances(),
                                             gm_data.get_inversed_covariances(), gm_data.get_amplitudes())
            loss = losses.sum()

            if self._logger:
                self._logger.log(iteration, losses, gm_data.pack_mixture())

        final_gm = gm_data.pack_mixture().float()
        final_gmm = gm.pack_mixture(gm_data.get_priors(), gm_data.get_positions(), gm_data.get_covariances())

        print("EM: # of invalid Gaussians: ", torch.sum(gm_data.get_priors() == 0).item())

        return final_gm, final_gmm

    def expectation(self, pcbatch_rep: torch.Tensor, gm_data) -> (torch.Tensor, torch.Tensor):
        batch_size = pcbatch_rep.shape[0]
        n_sample_points = pcbatch_rep.shape[2]
        oldpositions_rep = gm_data.get_positions().unsqueeze(2).expand(batch_size, 1, n_sample_points, self._n_gaussians, 3)
        oldicovs_rep = gm_data.get_inversed_covariances().unsqueeze(2).expand(batch_size, 1, n_sample_points, self._n_gaussians, 3, 3)
        grelpos = (pcbatch_rep - oldpositions_rep).unsqueeze(5) # bs, 1, np, ng, 3, 1
        gaussvalues = -0.5 * torch.matmul(grelpos.transpose(-2, -1), torch.matmul(oldicovs_rep, grelpos)).squeeze(
            5).squeeze(4) #bs, 1, np, ng
        priors_rep = torch.log(gm_data.get_amplitudes().unsqueeze(2).expand(batch_size, 1, n_sample_points, self._n_gaussians))
        likelihood_log = priors_rep + gaussvalues
        llh_sum = torch.logsumexp(likelihood_log, dim=3, keepdim=True) # bs, 1, np, ng
        losses = llh_sum.mean(dim = 2).squeeze()
        return torch.exp(likelihood_log - llh_sum), losses

    def maximization(self, pcbatch_rep: torch.Tensor, responsibilities: torch.Tensor, gm_data):
        # resp dim: (bs, 1, np, ng)
        batch_size = pcbatch_rep.shape[0]
        n_sample_points = pcbatch_rep.shape[2]
        n_k = responsibilities.sum(dim=2)  # dimension: (bs, 1, ng)
        # Calculate new Positions               # pcbatch_rep dimension: (bs, 1, np, ng, 3)
        multiplied = responsibilities.unsqueeze(4).expand_as(pcbatch_rep) * pcbatch_rep  # dimension: (bs, 1, np, ng, 3)
        new_positions = multiplied.sum(dim=2, keepdim=True) / n_k.unsqueeze(2).unsqueeze(
            4)  # dimension: (bs, 1, 1, ng, 3)
        new_positions_rep = new_positions.expand(batch_size, 1, n_sample_points, self._n_gaussians, 3)
        new_positions = new_positions.squeeze(2)  # dimensions: (bs, 1, ng, 3)
        # Calculate new Covariances
        relpos = (pcbatch_rep - new_positions_rep).unsqueeze(5)
        matrix = relpos * (relpos.transpose(-1, -2))  # dimension: (bs, 1, np, ng, 3, 3)
        matrix *= responsibilities.unsqueeze(4).unsqueeze(5)
        new_covariances = matrix.sum(dim=2) / n_k.unsqueeze(3).unsqueeze(4) + self._eps  # dimension: (b_s, 1, ng, 3, 3)
        # Calculate new priors
        new_priors = n_k / n_sample_points  # dimension: (b_s, 1, ng)
        new_positions[new_priors == 0] = torch.tensor([0.0, 0.0, 0.0]).double().cuda()
        new_covariances[new_priors == 0] = self._eps[0, 0, 0, :, :]
        # Update GMData
        gm_data.set_positions(new_positions)
        gm_data.set_covariances(new_covariances)
        gm_data.set_priors(new_priors)

    def initialize(self, pcbatch: torch.Tensor) -> torch.Tensor:
        batch_size = pcbatch.shape[0]
        point_count = pcbatch.shape[1]

        # Initialize according to "Finite Mixture Models", cahpter 2.12.2
        meanpos = pcbatch.mean(dim=1,keepdim=True) # bs, 1, 3
        diffs = (pcbatch - meanpos.expand(batch_size, point_count, 3)).unsqueeze(3) # bs, np, 3
        meanpos = meanpos.squeeze(1)
        meancov = (diffs * diffs.transpose(-1, -2)).mean(dim=[1])
        meanweight = 1.0 / self._n_gaussians

        positions = torch.zeros(batch_size, 1, self._n_gaussians, 3)
        for i in range(batch_size):
            positions[i, 0, :, :] = torch.tensor(
                numpy.random.multivariate_normal(meanpos[i, :].cpu(), meancov[i, :, :].cpu(), self._n_gaussians)).cuda()
        covariances = meancov.view(batch_size, 1, 1, 3, 3).expand(batch_size, 1, self._n_gaussians, 3, 3)
        weights = torch.zeros(batch_size, 1, self._n_gaussians)
        weights[:, :, :] = meanweight

        return gm.pack_mixture(weights.float().cuda(), positions.float().cuda(), covariances.float().cuda()).double()

    class TrainingData:

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
