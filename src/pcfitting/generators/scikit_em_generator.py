import numpy
import multiprocessing

from gmc import mat_tools
from pcfitting import GMMGenerator, GMLogger
from pcfitting import TerminationCriterion, MaxIterationTerminationCriterion
from pcfitting.error_functions import LikelihoodLoss
from pcfitting.generators.gmm_initializer import GMMInitializer
from .em_tools import EMTools
from .level_scaler import LevelScaler
import torch
import gmc.mixture as gm
from gmc import mat_tools
import math
import pcfitting.cpp.gmslib.src.pytorch_bindings.compute_mixture as gms
from sklearn.mixture import GaussianMixture


class ScikitEMGenerator(GMMGenerator):
    # Uses EM by SciKit-Learn

    def __init__(self,
                 n_components: int = 1,
                 tol: float = 1e-3,
                 reg_covar: float = 1e-6,
                 max_iter: int = 100,
                 n_init: int = 1,
                 init_params = 'kmeans',
                 alt_init = None,
                 ):
        self._n_components = n_components
        self._tol = tol
        self._reg_covar = reg_covar
        self._max_iter = max_iter
        self._n_init = n_init
        self._init_params = init_params
        self._alt_init = alt_init
        pass

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
        # gmbatch has to be None.
        # It returns two gaussian gmc.mixtures the first being a mixture with amplitudes as weights
        # the second a mixture where the weights describe the priors.
        # Training parameters have to be set in the other methods of the class


        batch_size = pcbatch.shape[0]
        assert (batch_size == 1), "ScikitEMGenerator currently does not support batchsizes > 1"
        point_count = pcbatch.shape[1]

        initweights = None
        initpositions = None
        initprecisions = None

        if gmbatch is not None:
            gmm = gm.convert_amplitudes_to_priors(gmbatch)
            initweights = gm.weights(gmm).numpy()
            initpositions = gm.positions(gmm).numpy()
            initprecisions = mat_tools.inverse(gm.covariances(gmm)).numpy()
        elif self._alt_init is not None:
            initializer = GMMInitializer(10000, 10000, torch.float32, self._reg_covar)
            gmbatch_init = initializer.initialize_by_method_name(self._alt_init, pcbatch,
                                                                 self._n_components, point_count, None, None)
            gmm = gm.convert_amplitudes_to_priors(gmbatch_init)
            initweights = gm.weights(gmm).numpy()
            initpositions = gm.positions(gmm).numpy()
            initprecisions = mat_tools.inverse(gm.covariances(gmm)).numpy()

        mixture = GaussianMixture(
            n_components=self._n_components,
            tol=self._tol,
            reg_covar=self._reg_covar,
            max_iter=self._max_iter,
            n_init=self._n_init,
            init_params=self._init_params,
            weights_init = initweights,
            means_init = initpositions,
            precisions_init= initprecisions,
            verbose=1
        )
        mixture.fit(pcbatch.view(-1, 3).cpu().numpy())
        positions = torch.tensor(mixture.means_, dtype=torch.float32, device=pcbatch.device).view(1, 1, self._n_components, 3)
        covariances = torch.tensor(mixture.covariances_, dtype=torch.float32, device=pcbatch.device).view(1, 1, self._n_components, 3, 3)
        weights = torch.tensor(mixture.weights_, dtype=torch.float32, device=pcbatch.device).view(1, 1, self._n_components)
        amplitudes = weights * gm.normal_amplitudes(covariances)

        gma = gm.pack_mixture(amplitudes, positions, covariances)
        gmm = gm.pack_mixture(weights, positions, covariances)

        return gma, gmm
