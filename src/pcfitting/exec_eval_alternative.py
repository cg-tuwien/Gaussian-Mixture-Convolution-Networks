from typing import List
from pcfitting import ErrorFunction, programs
from pcfitting.error_functions import LikelihoodLoss, PSNR, RMSE

# This takes up to two point clouds and uses them to evaluate all GMMS in one folder (+subdirectories)
# The results are printed to the console.

# --- CONFIGUREABLE VARIABLES ---
# Define Paths
fitpc_path = r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\data\dataset_vt_evaluation\fitpcs\n100000\bed_0003.off"
evalpc_path = r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\data\dataset_vt_evaluation\evalpcs\n100000\bed_0003.off"
gengmm_path = r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\data\dataset_vt_evaluation\gmms\DebugPreinerBed03"

gmaonly = True

error_functions: List[ErrorFunction] = [LikelihoodLoss(False), RMSE()]
error_function_identifiers = ["Likelihood Loss", "RMSE"]

# Scaling options
scaling_active = False
scaling_interval = (0, 1)

# --- DO NOT CHANGE FROM HERE ---

programs.execute_evaluation_singlepc_severalgm(fitpc_path, evalpc_path, gengmm_path,
                            error_functions, error_function_identifiers,
                            scaling_active, scaling_interval, gmaonly)