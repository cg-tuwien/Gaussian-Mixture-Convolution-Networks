from prototype_pcfitting import GMMGenerator, GMLogger, data_loading
from prototype_pcfitting import TerminationCriterion, MaxIterationTerminationCriterion
from prototype_pcfitting.error_functions import LikelihoodLoss
import torch
import gmc.mixture as gm


class EMGenerator(GMMGenerator):
    # GMM Generator using simple Expectation Maximization

    _device = torch.device('cuda')

    def __init__(self,
                 n_gaussians: int,
                 n_sample_points: int,
                 termination_criterion: TerminationCriterion = MaxIterationTerminationCriterion(500)):
        self._n_gaussians = n_gaussians
        self._n_sample_points = n_sample_points
        self._termination_criterion = termination_criterion
        self._logger = None

    def set_logging(self, logger: GMLogger = None):
        self._logger = logger

    def generate(self, pcbatch: torch.Tensor, gmbatch: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        self._termination_criterion.reset()

        batch_size = pcbatch.shape[0]
        point_count = pcbatch.shape[1]
        pcbatch = pcbatch.to(self._device)  # dimension: (bs, np, 3)
        pcbatch_4 = pcbatch.unsqueeze(1)  # dimension: (bs, 1, np, 3)

        lossgen = LikelihoodLoss()

        # Initialize mixture (Important: Assumes intervall [0,1])
        if gmbatch is None:
            gmbatch = gm.generate_random_mixtures(n_batch=batch_size, n_layers=1, n_components=self._n_gaussians,
                                                  n_dims=3, pos_radius=0.5,
                                                  cov_radius=0.01 / (self._n_gaussians ** (1 / 3)),
                                                  weight_min=0, weight_max=1, device=self._device)
            indizes = torch.randperm(point_count)[0:self._n_gaussians]
            positions = pcbatch[:, indizes, :].view(batch_size, 1, self._n_gaussians, 3)
            gmbatch = gm.pack_mixture(gm.weights(gmbatch), positions, gm.covariances(gmbatch))

        gm_data = self.TrainingData()
        gm_data.set_positions(gm.positions(gmbatch))
        gm_data.set_covariances(gm.covariances(gmbatch))
        gm_data.set_amplitudes(gm.weights(gmbatch))
        # responsibilities = torch.zeros((batch_size, point_count, self._n_gaussians))

        sample_points = data_loading.sample(pcbatch, self._n_sample_points)
        losses = lossgen.calculate_score(sample_points, gm_data.get_positions(), gm_data.get_covariances(),
                                         gm_data.get_inversed_covariances(), gm_data.get_amplitudes())
        loss = losses.sum()
        iteration = 0

        if self._logger:
            self._logger.log(iteration, losses, gm_data.pack_mixture())

        while self._termination_criterion.may_continue(iteration, loss.item()):
            iteration += 1

            # ToDo: Sometimes we can use unsqueeze instead of view, and maybe repeat is not necessary
            # ToDo: Erase NaNs, e.g. newCovariances = newCovariances + (newWeights < 0.0001).unsqueeze(-1).unsqueeze(-1) * torch.eye(n_dims, device=device).view(1, 1, 1, n_dims, n_dims) * 0.0001

            # Expectation
            gaussvalues = gm.evaluate_componentwise(gm_data.pack_mixture(), pcbatch_4)
            gaussvalues[gaussvalues < 1e-10] = 1e-10  # to avoid division by zero
            priors_rep = gm_data.get_priors().repeat(batch_size, 1, point_count, 1)
            multiplied = priors_rep * gaussvalues
            responsibilities = multiplied / (multiplied.sum(dim=3, keepdim=True) + 1e-20)   # dimension: (bs, 1, np, ng)

            # Maximization
            n_k = responsibilities.sum(dim=2)     # dimension: (bs, 1, ng)
            # Calculate new Positions               # pcbatch_rep dimension: (bs, 1, np, ng, 3)
            pcbatch_rep = pcbatch.unsqueeze(3).repeat(1, 1, 1, self._n_gaussians, 1)
            multiplied = responsibilities.unsqueeze(4).repeat(1, 1, 1, 1, 3) \
                            * pcbatch_rep  # dimension: (bs, 1, np, ng, 3)
            new_positions = multiplied.sum(dim=2, keepdim=True) / n_k.unsqueeze(2).unsqueeze(4)
                            # dimension: (bs, 1, 1, ng, 3)
            new_positions_rep = new_positions.repeat(1, 1, point_count, 1, 1)
            new_positions = new_positions.squeeze(2)  # dimensions: (bs, 1, ng, 3)
            # Calculate new Covariances
            relpos = (pcbatch_rep - new_positions_rep).unsqueeze(5)
            matrix = relpos * (relpos.transpose(-1, -2))    # dimension: (bs, 1, np, ng, 3, 3)
            matrix *= responsibilities.unsqueeze(4).unsqueeze(5)
            new_covariances = matrix.sum(dim = 2) / n_k.unsqueeze(3).unsqueeze(4)  # dimension: (b_s, 1, ng, 3, 3)
            # Calculate new priors
            new_priors = n_k / point_count  # dimension: (b_s, 1, ng)
            # Update GMData
            gm_data.set_positions(new_positions)
            gm_data.set_covariances(new_covariances)
            gm_data.set_priors(new_priors)

            sample_points = data_loading.sample(pcbatch, self._n_sample_points)
            losses = lossgen.calculate_score(sample_points, gm_data.get_positions(), gm_data.get_covariances(),
                                             gm_data.get_inversed_covariances(), gm_data.get_amplitudes())
            loss = losses.sum()

            if self._logger:
                self._logger.log(iteration, losses, gm_data.pack_mixture())

        final_gm = gm_data.pack_mixture()
        final_gmm = gm.pack_mixture(gm_data.get_priors(), gm_data.get_positions(), gm_data.get_covariances())

        return final_gm, final_gmm

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