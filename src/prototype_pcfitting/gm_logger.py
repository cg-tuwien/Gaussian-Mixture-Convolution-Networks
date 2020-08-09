import struct
from typing import List
import torch
import torch.utils.tensorboard
import os
import gmc.mixture as gm

from gmc.cpp.gm_vis import GMVisualizer, GmVisColorRangeMode, GmVisColoringRenderMode
from .scaler import Scaler


class GMLogger:

    def __init__(self,
                 names: List[str],
                 log_path: str,
                 log_positions: int = 0,
                 gm_n_components: int = 0,
                 log_loss_tb: int = 0,
                 log_rendering_tb: int = 0,
                 log_gm: int = 0,
                 pointclouds: torch.Tensor = None,
                 scaler: Scaler = None):

        # Prepare Tensorboard Log Data
        self._log_rendering_tb = log_rendering_tb
        self._log_loss_tb = log_loss_tb

        if log_rendering_tb > 0:
            self._visualizer = GMVisualizer(False, 500, 500)
            self._visualizer.set_camera_auto(True)
            if pointclouds:
                self._visualizer.set_pointclouds(pointclouds)
            self._visualizer.set_ellipsoid_coloring(GmVisColoringRenderMode.COLOR_WEIGHT,
                                                    GmVisColorRangeMode.RANGE_MINMAX)

        if log_rendering_tb > 0 or log_loss_tb > 0:
            self._tbwriters = []
            for i in range(len(names)):
                tbw = torch.utils.tensorboard.SummaryWriter(os.path.join(log_path, names[i]))
                self._tbwriters.append(tbw)

        # Prepare GM Log Data
        self._log_gm = log_gm
        self._gm_paths = [""] * len(names)
        if log_gm > 0 or log_positions > 0:
            for i in range(len(names)):
                n = names[i]
                self._gm_paths[i] = os.path.join(log_path, n)
                if not os.path.exists(self._gm_paths[i]):
                    os.mkdir(self._gm_paths[i])

        # Prepare Position Log Data
        self._log_positions = log_positions
        if log_positions > 0:
            self._gm_n_components = gm_n_components
            self._position_buffer = torch.zeros(len(names), gm_n_components, log_positions, 3)
            for i in range(len(names)):
                n = names[i]
                self._gm_paths[i] = os.path.join(log_path, n)
                for g in range(gm_n_components):
                    f = open(f"{self._gm_paths[i]}/pos-g{g}.bin", "w+")
                    f.close()

        self._scaler = scaler

    def log(self, iteration, losses, gmbatch):
        if self._log_loss_tb > 0 and iteration & self._log_loss_tb == 0:
            for i in range(len(self._tbwriters)):
                self._tbwriters[i].add_scalar("Loss", losses[i].item(), iteration)

        log_rendering = self._log_rendering_tb > 0 and iteration % self._log_rendering_tb == 0
        log_gm = self._log_gm > 0 and iteration % self._log_loss_tb == 0
        gm_upscaled = torch.zeros(0)

        if log_rendering or log_gm or self._log_positions > 0:
            gm_upscaled = self._scaler.scale_up_gm(gmbatch)

        if log_rendering:
            self._visualizer.set_gaussian_mixtures(gm_upscaled.detach().cpu(), isgmm=False)
            res = self._visualizer.render(iteration)
            for i in range(res.shape[0]):
                self._tbwriters[i].add_image(f"GM {i}, Ellipsoids", res[i, 0, :, :, :], iteration, dataformats="HWC")
                self._tbwriters[i].add_image(f"GM {i}, Density", res[i, 1, :, :, :], iteration, dataformats="HWC")
                self._tbwriters[i].flush()

        if log_gm:
            gmw = gm.weights(gm_upscaled)
            gmp = gm.positions(gm_upscaled)
            gmc = gm.covariances(gm_upscaled)
            for i in range(gm_upscaled.shape[0]):
                gm.write_gm_to_ply(gmw, gmp, gmc, i, f"{self._gm_paths[i]}/gmm-{str(iteration).zfill(5)}.ply")

        if self._log_positions > 0:
            self._position_buffer[:, :, (iteration-1) % self._log_positions, :] = \
                gm.positions(gm_upscaled).view(-1, self._gm_n_components, 3)

            if iteration & self._log_positions == 0:
                for i in range(len(self._gm_paths)):
                    for g in range(self._gm_n_components):
                        f = open(f"{self._gm_paths[i]}/pos-g{g}.bin", "a+b")
                        pdata = self._log_positions[i, g, :, :].view(-1)
                        bindata = struct.pack('<' + 'd'*len(pdata), *pdata)  # little endian!
                        f.write(bindata)
                        f.close()
