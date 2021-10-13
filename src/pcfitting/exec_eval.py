from typing import List
from pcfitting import EvalFunction, programs
from pcfitting.error_functions import LikelihoodLoss, PSNR, GMMStats, AvgDensities, ReconstructionStats, ReconstructionStatsProjected, Smoothness, ReconstructionStatsFiltered

# This takes a set of finished GMMs and evaluates them using several error functions.
# The results are printed to the console.

# --- CONFIGUREABLE VARIABLES ---
# Define Paths
# model_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_diff_scales/models"
# fitpc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_diff_scales/fitpcs"
# evalpc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_diff_scales/evalpcs"
# gengmm_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_diff_scales/gmms"
model_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_plane/models-0"
fitpc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_plane/fitpcs"
evalpc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_plane/evalpcs"
gengmm_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_plane/gmms"

# Define point count of pointclouds to use, and how many points to use for evaluation
n_points = 100000
pc2_points = 0

# Define identifiers of Generators to evaluate and error functions to use
generator_identifiers = ["SPf", "SPr", "SPb", "SPe", "HPf", "HPr", "HPb", "HPe"]

# Define Eval Functions
error_functions: List[EvalFunction] = [ReconstructionStatsProjected(ReconstructionStats(sample_points=100000)), AvgDensities()]

smallest_ev = None

# Scaling options
scaling_active = False
scaling_interval = (0, 1)

# Define directory name
# training_name = input('Name for training to evaluate: ')
# training_name = '210306-01-EmEckPre'
# training_name = '210312-EMepsvar'
# training_name = '210312-ng64'
# training_name = '210406-EmEckPre'
# training_name = '210529-SKEM'
# training_name = '210613-03-terminationEM'
# training_name = '210923-01-irr'
training_name = '211013-01-justcheckifstillworks'

# --- DO NOT CHANGE FROM HERE ---

programs.execute_evaluation(training_name, model_path, evalpc_path, None, gengmm_path, n_points, pc2_points,
                            generator_identifiers, error_functions, scaling_active, scaling_interval, smallest_ev)