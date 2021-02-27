import unittest
import torch
from gmc import mixture
from pcfitting import Scaler, ScalingMethod
from pcfitting.generators.level_scaler import LevelScaler


class ScalerTest(unittest.TestCase):

    def setUp(self):
        self._points = torch.tensor([[
            [1.0, 6.0, 9.0],
            [6.0, 2.0, 13.0],
            [11.0, 20.0, 3.0],
            [8.0, 22.0, 33.0],
            [2.0, 12.0, 18.0]
        ],[
            [0.0, 0.6, 0.3],
            [0.6, 0.2, 1.0],
            [1.0, 0.5, 0.3],
            [0.8, 1.0, 0.7],
            [0.2, 0.0, 0.0]
        ]], device='cuda')
        gmpositions = torch.tensor([[
            [1.0, 2.0, 3.0],
            [11.0, 22.0, 33.0]
        ], [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5]
        ]], device='cuda').view(2, 1, 2, 3)
        gmcovariances = torch.tensor([[
            [[0.4, 0.3, 0.2], [0.3, 0.6, 0.1], [0.2, 0.1, 0.9]],
            [[1.2, 0.3, 0.2], [0.3, 2.2, 0.1], [0.2, 0.1, 3.3]]
        ], [
            [[0.4, 0.3, 0.2], [0.3, 0.6, 0.1], [0.2, 0.1, 0.9]],
            [[1.2, 0.3, 0.2], [0.3, 2.2, 0.1], [0.2, 0.1, 3.3]]
        ]], device='cuda').view(2, 1, 2, 3, 3)
        assert(gmcovariances.det().gt(0).all())
        assert(gmcovariances[:, :, 0:2, 0:2].gt(0).all())
        gmpriors = torch.tensor([[[0.4, 0.6]], [[0.5, 0.5]]], device='cuda')
        gmamplitudes = gmpriors / (gmcovariances.det().sqrt() * 15.74960995)
        self._gm = mixture.pack_mixture(gmamplitudes, gmpositions, gmcovariances)
        self._gmm = mixture.pack_mixture(gmpriors, gmpositions, gmcovariances)

    def test_non_scaling(self):
        scaler = Scaler(active=False)
        scaler.set_pointcloud_batch(self._points)
        # Test PC
        scaledpc = scaler.scale_pc(self._points)
        unscaledpc = scaler.unscale_pc(scaledpc)
        self.assertTrue(self._points.eq(scaledpc).all())
        self.assertTrue(scaledpc.eq(unscaledpc).all())
        # Test GM
        scaledgm = scaler.scale_gm(self._gm)
        unscaledgm = scaler.unscale_gm(scaledgm)
        self.assertTrue(self._gm.eq(scaledgm).all())
        self.assertTrue(scaledgm.eq(unscaledgm).all())
        # Test GMM
        scaledgmm = scaler.scale_gmm(self._gmm)
        unscaledgmm = scaler.unscale_gmm(scaledgmm)
        self.assertTrue(self._gmm.eq(scaledgmm).all())
        self.assertTrue(scaledgmm.eq(unscaledgmm).all())

    def test_scaling_min_0_1(self):
        scaler = Scaler(active=True, interval=(0, 1), scaling_method=ScalingMethod.SMALLEST_SIDE_TO_MAX)
        scaler.set_pointcloud_batch(self._points)
        # Test PC
        scaledpc = scaler.scale_pc(self._points)
        unscaledpc = scaler.unscale_pc(scaledpc)
        scaledpc_should = torch.tensor([[
            [0.0, 0.4, 0.6],
            [0.5, 0.0, 1.0],
            [1.0, 1.8, 0.0],
            [0.7, 2.0, 3.0],
            [0.1, 1.0, 1.5]
        ],[
            [0.0, 0.6, 0.3],
            [0.6, 0.2, 1.0],
            [1.0, 0.5, 0.3],
            [0.8, 1.0, 0.7],
            [0.2, 0.0, 0.0]
        ]], device='cuda')
        self.assertTrue(torch.allclose(scaledpc, scaledpc_should))
        self.assertTrue(torch.allclose(unscaledpc, self._points))
        # Test GM
        scaledgm = scaler.scale_gm(self._gm)
        unscaledgm = scaler.unscale_gm(scaledgm)
        gmref_pos = torch.tensor([[
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0]
        ], [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5]
        ]], device='cuda').view(2, 1, 2, 3)
        gmref_cov = torch.tensor([[
            [[0.004, 0.003, 0.002], [0.003, 0.006, 0.001], [0.002, 0.001, 0.009]],
            [[0.012, 0.003, 0.002], [0.003, 0.022, 0.001], [0.002, 0.001, 0.033]]
        ], [
            [[0.4, 0.3, 0.2], [0.3, 0.6, 0.1], [0.2, 0.1, 0.9]],
            [[1.2, 0.3, 0.2], [0.3, 2.2, 0.1], [0.2, 0.1, 3.3]]
        ]], device='cuda').view(2, 1, 2, 3, 3)
        gmref_amp = mixture.weights(self._gmm) / (gmref_cov.det().sqrt() * 15.74960995)
        gmref = mixture.pack_mixture(gmref_amp, gmref_pos, gmref_cov)
        self.assertTrue(torch.allclose(gmref, scaledgm))
        self.assertTrue(torch.allclose(unscaledgm, self._gm))
        # Test GMM
        scaledgmm = scaler.scale_gmm(self._gmm)
        unscaledgmm = scaler.unscale_gmm(scaledgmm)
        gmmref = mixture.pack_mixture(mixture.weights(self._gmm), gmref_pos, gmref_cov)
        self.assertTrue(torch.allclose(gmmref, scaledgmm))
        self.assertTrue(torch.allclose(unscaledgmm, self._gmm))

    def test_scaling_max_m10_p10(self):
        scaler = Scaler(active=True, interval=(-10, 10), scaling_method=ScalingMethod.LARGEST_SIDE_TO_MAX)
        scaler.set_pointcloud_batch(self._points)
        # Test PC
        scaledpc = scaler.scale_pc(self._points)
        unscaledpc = scaler.unscale_pc(scaledpc)
        scaledpc_should = torch.tensor([[
            [-10.0000, -7.33333, -6.00000],
            [-6.66666, -10.0000, -3.33333],
            [-3.33333, 2.000000, -10.0000],
            [-5.33333, 3.333333, 10.00000],
            [-9.33333, -3.33333, 0.000000]
        ], [
            [-10.0, 2.0, -4.0],
            [2.0, -6.0, 10.0],
            [10.0, 0.0, -4.0],
            [6.0, 10.0, 4.0],
            [-6.0, -10.0, -10.0]
        ]], device='cuda')
        self.assertTrue(torch.allclose(scaledpc, scaledpc_should))
        self.assertTrue(torch.allclose(unscaledpc, self._points))
        # Test GM
        scaledgm = scaler.scale_gm(self._gm)
        unscaledgm = scaler.unscale_gm(scaledgm)
        gmref_pos = torch.tensor([[
            [-10.0, -10.0, -10.0],
            [-3.33333333333333333, 3.33333333333333333, 10.0]
        ], [
            [-10.0, -10.0, -10.0],
            [0.0, 0.0, 0.0]
        ]], device='cuda').view(2, 1, 2, 3)
        gmref_cov = torch.tensor([[
            [[0.4, 0.3, 0.2], [0.3, 0.6, 0.1], [0.2, 0.1, 0.9]],
            [[1.2, 0.3, 0.2], [0.3, 2.2, 0.1], [0.2, 0.1, 3.3]]
        ], [
            [[160.0, 120.0, 80.0], [120.0, 240.0, 40.0], [80.0, 40.0, 360.0]],
            [[480.0, 120.0, 80.0], [120.0, 880.0, 40.0], [80.0, 40.0, 1320.0]]
        ]], device='cuda').view(2, 1, 2, 3, 3)
        gmref_cov[0] /= (1.5 ** 2)
        gmref_amp = mixture.weights(self._gmm) / (gmref_cov.det().sqrt() * 15.74960995)
        gmref = mixture.pack_mixture(gmref_amp, gmref_pos, gmref_cov)
        self.assertTrue(torch.allclose(gmref, scaledgm))
        self.assertTrue(torch.allclose(unscaledgm, self._gm))
        # Test GMM
        scaledgmm = scaler.scale_gmm(self._gmm)
        unscaledgmm = scaler.unscale_gmm(scaledgmm)
        gmmref = mixture.pack_mixture(mixture.weights(self._gmm), gmref_pos, gmref_cov)
        self.assertTrue(torch.allclose(gmmref, scaledgmm))
        self.assertTrue(torch.allclose(unscaledgmm, self._gmm))

    def test_level_scaler_inactive(self):
        pcbatch = torch.tensor([[
            [1.0, 1.2, 0.8],
            [2.1, 2.8, 3.4],
            [0.9, 1.8, 3.0],
            [1.9, 2.9, 3.0],
            [0.9, 2.4, 2.5],
            [9.9, 9.8, 9.7]
        ]]).cuda()
        parent_per_point = torch.tensor([[0, 0, 0, 1, 1, 2]])
        scaler = LevelScaler(active=False)
        scaler.set_pointcloud(pcbatch, parent_per_point, 3)
        scaled_pc = scaler.scale_pc(pcbatch)
        self.assertTrue(torch.allclose(scaled_pc, pcbatch))

        gmpositions = torch.tensor([[[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ]]]).cuda()
        gmcovariances = torch.eye(3, 3).view(1, 1, 1, 3, 3).expand(1, 1, 9, 3, 3).cuda()
        gmweights = torch.zeros(1, 1, 9).cuda()
        gmweights[:, :, :] = 1 / 3.0

        gmwn, gmpn, gmcn = scaler.unscale_gmm_wpc(gmweights, gmpositions, gmcovariances)
        self.assertTrue(torch.allclose(gmpn, gmpositions))
        self.assertTrue(torch.allclose(gmcn, gmcovariances))
        self.assertTrue(torch.allclose(gmwn, gmweights))

    def test_level_scaler_active(self):
        pcbatch = torch.tensor([[
            [1.0, 1.2, 0.8],
            [2.1, 2.8, 3.4],
            [0.9, 1.8, 3.0],
            [1.9, 2.9, 3.0],
            [0.9, 2.4, 2.5],
            [9.9, 9.8, 9.7]
        ]]).cuda()
        parent_per_point = torch.tensor([[0, 0, 0, 1, 1, 2]])
        scaler = LevelScaler(active=True, interval=(0.0, 1.0))
        scaler.set_pointcloud(pcbatch, parent_per_point, 4)
        scaled_pc = scaler.scale_pc(pcbatch)
        scaled_pc_should = torch.tensor([[
            [0.03846153, 0, 0],
            [0.46153846, 0.615384615, 1.0],
            [0, 0.2307692307, 0.846153846],
            [1, 0.5, 0.5],
            [0, 0, 0],
            [0.0, 0.0, 0.0]
        ]], device='cuda')
        self.assertTrue(torch.allclose(scaled_pc, scaled_pc_should))

        gmpositions = torch.tensor([[[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]]]).cuda()
        gmcovariances = torch.eye(3, 3).view(1, 1, 1, 3, 3).expand(1, 1, 16, 3, 3).cuda()
        gmweights = torch.zeros(1, 1, 16).cuda()
        gmweights[:, :, :] = 1 / 4.0

        gmwn, gmpn, gmcn = scaler.unscale_gmm_wpc(gmweights, gmpositions, gmcovariances)
        gmpn_should = torch.tensor([[[
            [0.9, 1.2, 0.8],
            [3.5, 1.2, 0.8],
            [3.5, 3.8, 3.4],
            [3.5, 3.8, 3.4],
            [0.9, 2.4, 2.5],
            [1.9, 2.4, 2.5],
            [1.9, 3.4, 3.5],
            [1.9, 3.4, 3.5],
            [9.9, 9.8, 9.7],
            [10.9, 9.8, 9.7],
            [10.9, 10.8, 10.7],
            [10.9, 10.8, 10.7],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ]]], device='cuda')
        gmcn_should = gmcovariances.clone()
        gmcn_should[0, 0, 0:4] *= (2.6**2)
        self.assertTrue(torch.allclose(gmpn, gmpn_should))
        self.assertTrue(torch.allclose(gmcn, gmcn_should))
        self.assertTrue(torch.allclose(gmwn, gmweights))

    def test_level_scaler_active_neg(self):
        pcbatch = torch.tensor([[
            [1.0, 1.2, 0.8],
            [2.1, 2.8, 3.4],
            [0.9, 1.8, 3.0],
            [1.9, 2.9, 3.0],
            [0.9, 2.4, 2.5],
            [9.9, 9.8, 9.7]
        ]]).cuda()
        parent_per_point = torch.tensor([[0, 0, 0, 1, 1, 2]])
        scaler = LevelScaler(active=True, interval=(-2.0, 0.0))
        scaler.set_pointcloud(pcbatch, parent_per_point, 3)
        scaled_pc = scaler.scale_pc(pcbatch)
        scaled_pc_should = torch.tensor([[
            [0.03846153, 0, 0],
            [0.46153846, 0.615384615, 1.0],
            [0, 0.2307692307, 0.846153846],
            [1, 0.5, 0.5],
            [0, 0, 0],
            [0.0, 0.0, 0.0]
        ]], device='cuda') * 2 - 2
        self.assertTrue(torch.allclose(scaled_pc, scaled_pc_should))

        gmpositions = torch.tensor([[[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ]]]).cuda() * 2 - 2
        gmcovariances = torch.eye(3, 3).view(1, 1, 1, 3, 3).expand(1, 1, 9, 3, 3).cuda()
        gmweights = torch.zeros(1, 1, 9).cuda()
        gmweights[:, :, :] = 1 / 3.0

        gmwn, gmpn, gmcn = scaler.unscale_gmm_wpc(gmweights, gmpositions, gmcovariances)
        gmpn_should = torch.tensor([[[
            [0.9, 1.2, 0.8],
            [3.5, 1.2, 0.8],
            [3.5, 3.8, 3.4],
            [0.9, 2.4, 2.5],
            [1.9, 2.4, 2.5],
            [1.9, 3.4, 3.5],
            [9.9, 9.8, 9.7],
            [11.9, 9.8, 9.7],
            [11.9, 11.8, 11.7]
        ]]], device='cuda')
        gmcn_should = gmcovariances.clone()
        gmcn_should[0, 0, 0:3] *= (1.3 ** 2)
        gmcn_should[0, 0, 3:6] *= (0.5 ** 2)
        self.assertTrue(torch.allclose(gmpn, gmpn_should))
        self.assertTrue(torch.allclose(gmcn, gmcn_should))
        self.assertTrue(torch.allclose(gmwn, gmweights))


if __name__ == '__main__':
    unittest.main()
