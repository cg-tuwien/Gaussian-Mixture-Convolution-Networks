from typing import List
from prototype_pcfitting import ErrorFunction, programs
from prototype_pcfitting.error_functions import LikelihoodLoss

# This takes a set of finished GMMs and evaluates them using several error functions.
# The results are printed to the console.

# --- CONFIGUREABLE VARIABLES ---
# Define Paths
model_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/models"
genpc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/pointclouds"
gengmm_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/gmms"

# Define point count of pointclouds to use, and how many points to use for evaluation
n_points = 1000
eval_points = 1000

# Define identifiers of Generators to evaluate and error functions to use
#generator_identifiers = ["EMi0", "EMi3", "EckHP", "EckSP0.3", "EckSP0.5"]
#generator_identifiers = ["EckSP0.0"]
# generator_identifiers = ["00-EMiRNP", "01-EMiFspmax", "10-EckHPiTBB", "11-EckHPiRNP", "12-EckHPiFpsmax", "13-EckHPiKM",
#                          "14-EckHPiEigen", "20-EckSP0.0iBB","21-EckSP0.0iRNP", "22-EckSP0.0iFpsmax", "23-EckSP0.0iKM",
#                          "30-EckSP0.1iBB","31-EckSP0.1iRNP", "32-EckSP0.1iFpsmax", "33-EckSP0.1iKM",
#                          "40-EckSP0.3iBB","41-EckSP0.3iRNP", "42-EckSP0.3iFpsmax", "43-EckSP0.3iKM"]
generator_identifiers = ["10-EckHPiTBB(unscaled)", "11-EckHPiRNP(unscaled)", "14-EckHPiEigen(unscaled)"]
error_functions: List[ErrorFunction] = [] # [LikelihoodLoss()]
error_function_identifiers = [] # ["Likelihood Loss"]

# Scaling options
scaling_active = False
scaling_interval = (0, 1)

# --- DO NOT CHANGE FROM HERE ---
# Read in Name
training_name = input('Name for training to evaluate: ')

programs.execute_evaluation(training_name, model_path, genpc_path, gengmm_path, n_points, eval_points,
                            generator_identifiers, error_functions, error_function_identifiers,
                            scaling_active, scaling_interval)