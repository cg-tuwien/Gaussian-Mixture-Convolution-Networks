import numpy

from gmc import mat_tools
from prototype_pcfitting import GMMGenerator, GMLogger
from prototype_pcfitting import TerminationCriterion, MaxIterationTerminationCriterion
from prototype_pcfitting.error_functions import LikelihoodLoss
from .em_tools import EMTools
from .level_scaler import LevelScaler
import torch
import gmc.mixture as gm
from gmc import mat_tools
import math
import prototype_pcfitting.cpp.gmslib.src.pytorch_bindings.compute_mixture as gms


class PreinerGenerator(GMMGenerator):

    def __init__(self,
                 initNeighborhoodType: int = 1,
                 knnCount: int = 8,
                 maxInitNeighborDist: float = 1.0,
                 initIsotropicStdev: float = 1.0,
                 initIsotropic: bool = False,
                 useWeightedPotentials: bool = False,
                 initMeansInPoints: bool = True,
                 nLevels: int = 4,
                 alpha: float = 2.2,
                 fixedNumberOfGaussians: int = 0):
        # Creates a new PreinerGenerator
        # Parameters:
        #   initNeighborhoodType: int
        #       0: initialize Gaussian using all samples within maxInitNeighborDist,
        #       1: use only kNNCount nearest neighbors.
        #   knnCount: int
        #       number nearest neighbors per point used for initial Gaussian computation.
        #       The neighbor set is clamped by maxInitNeighborDist.
        #   maxInitNeighborDist: float
        #       global initialization kernel radius, in % of to BB-Size
        #   initIsotropicStdev: float
        #       if isotropic initialization is active, the initial standard deviation of the Gaussians in % of BB-Size
        #   initIsotropic: bool
        #       isotropic initial Gaussians, with stddev initIsotropicStdev
        #   useWeightedPotentials: bool
        #       if true, performs WLOP-like balancing of the initial Gaussian potentials
        #   initMeansInPoints: bool
        #       positions the initial Gaussians in the point positions instead of the local means
        #   nLevels: int
        #       number of levels to use when clustering
        #   alpha: float
        #       multiple of cluster maximum std deviation to use for query radius
        #   fixedNumberOfGaussians: int
        #       If 0, the number of Gaussians is not determined in advance.
        #       Otherwise this will be the amount of Gaussians in the result. If activated, nLevels will be ignored
        #
        self._params = gms.Params()
        self._params.verbose = True
        self._params.initNeighborhoodType = initNeighborhoodType
        self._params.kNNCount = knnCount
        self._params.maxInitNeighborDist = maxInitNeighborDist
        self._params.initIsotropicStdev = initIsotropicStdev
        self._params.initIsotropic = initIsotropic
        self._params.useWeightedPotentials = useWeightedPotentials
        self._params.initMeansInPoints = initMeansInPoints
        self._params.nLevels = nLevels
        self._params.alpha = alpha
        self._params.fixedNumberOfGaussians = fixedNumberOfGaussians

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

        assert (gmbatch is None), "PreinerGenerator cannot improve existing GMMs"

        batch_size = pcbatch.shape[0]
        assert (batch_size is 1), "PreinerGenerator currently does not support batchsizes > 1"
        point_count = pcbatch.shape[1]

        gmm = gms.compute_mixture(pcbatch[0], self._params).view(1, 1, -1, 13).cuda()
        gma = gm.convert_priors_to_amplitudes(gmm)

        if self._logger:
            loss = LikelihoodLoss(False).calculate_score_packed(pcbatch, gma)
            self._logger.log(0, loss, gma)
            self._logger.log(100, loss, gma)

        print("Number of Gaussians: ", gma.shape[2])

        return gma, gmm
