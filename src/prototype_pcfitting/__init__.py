import os
import sys
source_dir = os.path.dirname(__file__)
sys.path.append(source_dir + '/..')

from prototype_pcfitting.gmm_generator import GMMGenerator
from prototype_pcfitting.error_function import ErrorFunction