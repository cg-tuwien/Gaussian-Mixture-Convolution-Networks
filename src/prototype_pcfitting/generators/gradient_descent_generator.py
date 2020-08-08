from typing import Tuple

from prototype_pcfitting import GMMGenerator
from prototype_pcfitting.error_functions import LikelihoodLoss
import prototype_pcfitting.pointcloud as pointcloud
import torch
import torch.optim
import torch.optim.rmsprop
import gmc.mixture as gm


class GradientDescentGenerator(GMMGenerator):

    _device = torch.device('cuda')

    def __init__(self,
                 n_components: int,
                 n_sample_points: int,
                 learn_rate_pos: float = 1e-3,
                 learn_rate_cov: float = 1e-4,
                 learn_rate_weights: float = 5e-4):
        self._n_components = n_components
        self._n_sample_points = n_sample_points
        self._learn_rate_pos = learn_rate_pos
        self._learn_rate_cov = learn_rate_cov
        self._learn_rate_weights = learn_rate_weights
        self._loss = LikelihoodLoss()

    def generate(self, pcbatch: torch.Tensor, gmbatch: torch.Tensor = None) -> torch.Tensor:
        batch_size = pcbatch.shape[0]
        point_count = pcbatch.shape[1]

        # Initialize mixture
        create_new_mixture = gmbatch is None
        if create_new_mixture:
            gmbatch = gm.generate_random_mixtures(n_batch=batch_size, n_layers=1, n_components=self._n_components,
                                                  n_dims=3, pos_radius=0.5,
                                                  cov_radius=0.01 / (self._n_components**(1/3)),
                                                  weight_min=0, weight_max=1, device=self._device)
            indizes = torch.randperm(point_count)[0:self._n_components]
            positions = pcbatch[:, indizes, :].view(batch_size, 1, self._n_components, 3) - 0.5
            gmbatch = gm.pack_mixture(gm.weights(gmbatch), positions, gm.covariances(gmbatch))

        # TODO: Scaling is still considered here! Creation is done in [0,1], not in pcsize

        # Initialize Training Data
        gm_data = self.TrainingData()
        gpositions = gm.positions(gmbatch)
        gpositions += 0.5
        gm_data.set_positions(gpositions)
        gm_data.set_covariances(gm.covariances(gmbatch))
        gm_data.set_amplitudes(gm.weights(gmbatch))

        # Initialize Optimizers
        optimiser_pos = torch.optim.rmsprop.RMSprop([gm_data.tr_positions], lr=self._learn_rate_pos, alpha=0.7, momentum=0.0)
        optimiser_cov = torch.optim.Adam([gm_data.tr_cov_data], lr=self._learn_rate_cov)
        optimiser_pi = torch.optim.Adam([gm_data.tr_pi_relative], lr=self._learn_rate_weights)

        # TODO: Tensor Board Writer and Logging

        while True:
            optimiser_pos.zero_grad()
            optimiser_cov.zero_grad()
            optimiser_pi.zero_grad()

            sample_points = pointcloud.sample(pcbatch, self._n_sample_points)
            loss = self._loss.calculate_score(sample_points, gm_data.get_positions(), gm_data.get_covariances(),
                                              gm_data.get_inversed_covariances(), gm_data.get_amplitudes())
            loss.backward()
            optimiser_pos.step()
            optimiser_cov.step()
            optimiser_pi.step()
            gm_data.update_covariances()
            gm_data.update_amplitudes()

            # TODO: Abbruchbedingung

        return gm.pack_mixture(gm_data.get_amplitudes(), gm_data.get_positions(), gm_data.get_covariances())

    class TrainingData:

        def __init__(self):
            self.tr_positions = None
            self.tr_cov_data = None
            self.tr_pi_relative = None
            self.pi_normalized = None
            self.pi_sum = None
            self.covariances = None
            self.inversed_covariances = None
            self.determinants = None
            self.amplitudes = None
            self._epsilon = pow(10, -2.6)

        def set_positions(self, positions: torch.Tensor):
            self.tr_positions = positions
            self.tr_positions.requires_grad = True

        def set_covariances(self,  covariances: torch.Tensor):
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
            batch_size = amplitudes.shape[0]
            self.tr_pi_relative = amplitudes * self.determinants.sqrt() * 15.74960995
            self.tr_pi_relative.requires_grad = True
            self.update_amplitudes()

        def update_covariances(self):
            cov_shape = self.tr_cov_data.shape
            cov_factor_mat_rec = torch.zeros((cov_shape[0], cov_shape[1], cov_shape[2], 3, 3)).to(GradientDescentGenerator._device)
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
            self.determinants = torch.pow(cov_factor_mat_rec[:, :, :, 0, 0] * cov_factor_mat_rec[:, :, :, 1, 1] \
                                     * cov_factor_mat_rec[:, :, :, 2, 2], 2)

        def update_amplitudes(self):
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