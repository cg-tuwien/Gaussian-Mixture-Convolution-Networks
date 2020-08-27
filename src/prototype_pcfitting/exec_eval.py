from typing import List
from prototype_pcfitting import ErrorFunction, programs
from prototype_pcfitting.error_functions import LikelihoodLoss

# This takes a set of finished GMMs and evaluates them using several error functions.
# The results are printed to the console.

# --- CONFIGUREABLE VARIABLES ---
# Define Paths
model_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/models"
genpc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/pointclouds"
gengmm_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/gmms"

# Define point count of pointclouds to use, and how many points to use for evaluation
n_points = 20000
eval_points = 20000

# Define identifiers of Generators to evaluate and error functions to use
generator_identifiers = ["GD1", "GD2"]
error_functions: List[ErrorFunction] = [LikelihoodLoss()]
error_function_identifiers = ["Likelihood Loss"]

# --- DO NOT CHANGE FROM HERE ---
# Read in Name
training_name = input('Name for training to evaluate: ')

programs.execute_evaluation(training_name, model_path, genpc_path, gengmm_path, n_points, eval_points,
                            generator_identifiers, error_functions, error_function_identifiers)