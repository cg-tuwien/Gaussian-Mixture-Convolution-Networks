import numpy
import multiprocessing

from gmc import mat_tools
from pcfitting import GMMGenerator, GMLogger
from pcfitting import TerminationCriterion, MaxIterationTerminationCriterion
from pcfitting.error_functions import LikelihoodLoss
from .em_tools import EMTools
from .level_scaler import LevelScaler
import torch
import gmc.mixture as gm
from gmc import mat_tools
import math
import pcfitting.cpp.gmslib.src.pytorch_bindings.compute_mixture as gms


class PreinerGenerator(GMMGenerator):
    # Generator for Geometrically regularized bottom-up-HEM by Preiner et al.

    def __init__(self,
                 alpha: float = 2.0,
                 pointpos: bool = True,
                 stdev: float = 0.01,
                 iso: bool = False,
                 inittype: str = "fixed",
                 knn: int = 8,
                 fixeddist: float = 0.1,
                 weighted: bool = False,
                 levels: int = 20,
                 reductionfactor: float = 3,
                 ngaussians: int = 0,
                 avoidorphansmode: int = 0,
                 verbosity: int = 2):
        # Creates a new PreinerGenerator
        # Parameters:
        #   alpha: float
        #       Clustering regularization parameter
        #   pointpos: bool
        #       Initializes Gaussian positions in point locations rather than local point means
        #   stdev: float
        #       Default isotropic standard deviation bias of each initial Gaussian [in %bbd]
        #   iso: bool
        #       Initialize mixture with isotropic Gaussians of standard deviation <stdev>
        #   inittype: str
        #       'knn' - Init anisotropic Gaussians based on KNN; 'fixed' - based on fixed distance
        #   knn: int
        #       Number of nearest neighbors considered for 'knn' initialization
        #   fixeddist: float
        #       Max neighborhood distance for points considered for 'fixed' initialization [in %bbd]
        #   weighted: bool
        #       Initializes mixture with locally normalized density
        #   levels: int
        #       Number of HEM clustering levels
        #   reductionFactor: float
        #       HEM Reduction Factor
        #   ngaussians: int
        #       Fixed Number of output Gaussians if desired, otherwise zero. If not zero, levels will be ignored, and
        #       chosen automatically.
        #   avoidorphans: int
        #       0: Regular Mode
        #       1: Assigns each Child Gaussian to at least one parent
        #       2: As mode 0 but if convergence to desired number of gaussians is not possible, switches to one
        # Quantities described with '[in %bbd]' are given in percent of the input point cloud bounding box diagonal.
        #
        self._params = gms.Params()
        self._params.alpha = alpha
        self._params.pointpos = pointpos
        self._params.stdev = stdev
        self._params.iso = iso
        self._params.inittype = inittype
        self._params.knn = knn
        self._params.fixeddist = fixeddist
        self._params.weighted = weighted
        self._params.levels = levels
        self._params.reductionfactor = reductionfactor
        self._params.ngaussians = ngaussians
        self._params.avoidorphans = avoidorphansmode
        self._params.verbose = verbosity >= 2
        self._params.threads = multiprocessing.cpu_count()
        self._logger = None

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
        assert (batch_size == 1), "PreinerGenerator currently does not support batchsizes > 1"
        point_count = pcbatch.shape[1]

        gmm = gms.compute_mixture(pcbatch[0], self._params).view(1, 1, -1, 13).cuda()
        gma = gm.convert_priors_to_amplitudes(gmm)

        if self._logger:
            loss = LikelihoodLoss(False).calculate_score_packed(pcbatch, gma)
            self._logger.log(0, loss, gma)
            self._logger.log(100, loss, gma)

        if self._params.verbose:
            print("Number of Gaussians: ", gma.shape[2])

        if self._params.ngaussians != gma.shape[2]:
            print("Warning: number of Gaussians not correct!")

        return gma, gmm
