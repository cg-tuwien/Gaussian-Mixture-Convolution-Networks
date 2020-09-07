import struct
from typing import List
import torch
import torch.utils.tensorboard
import os
import gmc.mixture as gm

from gmc.cpp.gm_visualizer import GMVisualizer, GmVisColorRangeMode, GmVisColoringRenderMode
from .scaler import Scaler
from prototype_pcfitting import data_loading


class GMLogger:
    # Class that provides logging functionality.
    # This class may log loss to the console, loss and renderings to tensorboard,
    # mixtures to ply-fiels and positions to binary files (can be read by the visualizer).
    # Please take note, that only one GMLogger with visualization functionality can exist at once.
    # If you stop using one, please call finalize, in order to enable a new one to work properly.

    def __init__(self,
                 names: List[str],
                 log_prefix: str,
                 log_path: str,
                 log_positions: int = 0,
                 gm_n_components: int = 0,
                 log_loss_console: int = 0,
                 log_loss_tb: int = 0,
                 log_rendering_tb: int = 0,
                 log_gm: int = 0,
                 pointclouds: torch.Tensor = None,
                 scaler: Scaler = None):
        # Constructor. Parameters:
        #   names: List[str]
        #       List of identifiers. There should be a identifier for each batch entry.
        #       These are used as directory names for the logging
        #   log_prefix: str
        #       A prefix to prepend to the identifiers
        #   log_path: str
        #       Root directory for the logs.
        #   log_positions: int
        #       If positions should be logged, this number identifies after how many iterations
        #       the positions should be logged to the disk. A lower positive number increases execution time.
        #       0 means no position logging.
        #   gm_n_components: int
        #       The number of Gaussians per GM. Needs to be given for position logging
        #   log_loss_console: int
        #       If the loss should be logged ot the console, this number identifies after how many iterations
        #       the current loss should be logged. 0 means no logging.
        #   log_loss_tb: int
        #       If the loss should be logged to tensorboard, this number identifies after how many iterations
        #       the current loss should be logged. Ideally, this is 1. 0 means no loss logging.
        #   log_rendering_tb: int
        #       If the visualizations should be logged to tensorboard, this number identifies after how many
        #       iterations a rendering be logged. A lower positive number increases execution time.
        #       0 means no visualization logging.
        #   log_gm: int
        #       If the mixtures should be logged to disk, this number identifies after how many iterations
        #       the gms are logged. 0 means no gm logging.
        #   pointclouds: torch.Tensor
        #       If the tensorboard renderings should contain pointclouds, this variable should be set with the
        #       corresponding point cloud batch. (in original scale)
        #   scaler: Scaler
        #       Needs to be set for everything except loss logging. The scaler for upscaling the given GMMs.

        # Prepare Tensorboard Log Data
        self._log_rendering_tb = log_rendering_tb
        self._log_loss_tb = log_loss_tb
        self._log_loss_console = log_loss_console
        self._names = names

        if log_rendering_tb > 0:
            self._visualizer = GMVisualizer(False, 500, 500)
            self._visualizer.set_camera_auto(True)
            if pointclouds is not None:
                self._visualizer.set_pointclouds(pointclouds.cpu())
            self._visualizer.set_ellipsoid_coloring(GmVisColoringRenderMode.COLOR_WEIGHT,
                                                    GmVisColorRangeMode.RANGE_MINMAX)

        if log_rendering_tb > 0 or log_loss_tb > 0:
            self._tbwriters = []
            for i in range(len(names)):
                tbw = torch.utils.tensorboard.SummaryWriter(os.path.join(log_path, log_prefix, names[i]))
                self._tbwriters.append(tbw)

        # Prepare GM Log Data
        self._log_gm = log_gm
        self._gm_paths = [""] * len(names)
        if log_gm > 0 or log_positions > 0:
            for i in range(len(names)):
                n = names[i]
                self._gm_paths[i] = os.path.join(log_path, log_prefix, n)
                if not os.path.exists(self._gm_paths[i]):
                    os.makedirs(self._gm_paths[i])

        # Prepare Position Log Data
        self._log_positions = log_positions
        if log_positions > 0:
            self._gm_n_components = gm_n_components
            self._position_buffer = torch.zeros(len(names), gm_n_components, log_positions, 3)
            for i in range(len(names)):
                n = names[i]
                self._gm_paths[i] = os.path.join(log_path, log_prefix, n)
                for g in range(gm_n_components):
                    f = open(f"{self._gm_paths[i]}/pos-g{g}.bin", "w+")
                    f.close()

        self._scaler = scaler

    def log(self, iteration: int, losses: torch.Tensor, gmbatch: torch.Tensor):
        # Performs logging.
        # Parameters:
        #   iteration: int
        #       Current iteration number (important, as some logging might only happen every nth iteration)
        #   losses: torch.Tensor
        #       List of losses (necessary if loss logging is active)
        #   gmbatch: torch.Tensor
        #       Current Gaussians (necessary for everything except loss logging)
        if self._log_loss_console:
            for b in range(len(self._names)):
                print(f"Iteration {iteration}. Loss of GM {self._names[b]}: {losses[b]}")

        if self._log_loss_tb > 0 and iteration & self._log_loss_tb == 0:
            for i in range(len(self._tbwriters)):
                self._tbwriters[i].add_scalar("Loss", losses[i].item(), iteration)
                self._tbwriters[i].flush()

        log_rendering = self._log_rendering_tb > 0 and iteration % self._log_rendering_tb == 0
        log_gm = self._log_gm > 0 and iteration % self._log_gm == 0
        gm_upscaled = torch.zeros(0)

        if log_rendering or log_gm or self._log_positions > 0:
            gm_upscaled = self._scaler.scale_up_gm(gmbatch)

        if log_rendering:
            self._visualizer.set_gaussian_mixtures(gm_upscaled.detach().cpu(), isgmm=False)
            res = self._visualizer.render(iteration)
            for i in range(res.shape[0]):
                self._tbwriters[i].add_image(f"Ellipsoids", res[i, 0, :, :, :], iteration, dataformats="HWC")
                self._tbwriters[i].add_image(f"Density", res[i, 1, :, :, :], iteration, dataformats="HWC")
                self._tbwriters[i].flush()

        if log_gm:
            gmw = gm.weights(gm_upscaled)
            gmp = gm.positions(gm_upscaled)
            gmc = gm.covariances(gm_upscaled)
            for i in range(gm_upscaled.shape[0]):
                data_loading.write_gm_to_ply(gmw, gmp, gmc, i,
                                            f"{self._gm_paths[i]}/gmm-{str(iteration).zfill(5)}.gma.ply")

        if self._log_positions > 0:
            self._position_buffer[:, :, iteration % self._log_positions, :] = \
                gm.positions(gm_upscaled).view(-1, self._gm_n_components, 3)

            if (iteration+1) % self._log_positions == 0:
                for i in range(len(self._gm_paths)):
                    for g in range(self._gm_n_components):
                        f = open(f"{self._gm_paths[i]}/pos-g{g}.bin", "a+b")
                        pdata = self._position_buffer[i, g, :, :].view(-1)
                        bindata = struct.pack('<' + 'd'*len(pdata), *pdata)  # little endian!
                        f.write(bindata)
                        f.close()

    def finalize(self):
        # This has to be called when using the visualizer before creating a new GMLogger!
        if self._log_loss_tb > 0 or self._log_rendering_tb > 0:
            for i in range(len(self._tbwriters)):
                self._tbwriters[i].close()
        if self._log_rendering_tb > 0:
            self._visualizer.finish()
