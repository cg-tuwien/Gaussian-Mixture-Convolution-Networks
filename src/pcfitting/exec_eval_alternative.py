from typing import List
from pcfitting import EvalFunction, programs
from pcfitting.error_functions import GMMStats, Smoothness, AvgDensities, ReconstructionStats, ReconstructionStatsProjected, ReconstructionStatsFiltered

# This takes up to two point clouds and uses them to evaluate all GMMS in one folder (+subdirectories)
# The results are printed to the console.

# --- CONFIGUREABLE VARIABLES ---
# Define Paths
#fitpc_path = r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\data\dataset_diff_scales\fitpcs\n100000\bed_0001.off"
evalpc_path = r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\data\dataset_diff_scales\evalpcs\n100000\bed_0001.off"
#evalpc_path = r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\data\dataset_plane\evalpcs\n100000\plane0-original.off"
gengmm_path = r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\data\dataset_diff_scales\bedcollection"
#gengmm_path = r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\data\dataset_plane\gmms\211003-03-thesis"
model_path = r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\data\dataset_diff_scales\models\bed_0001.off"
#model_path = r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\data\dataset_plane\models-0\plane0-original.off"

gmaonly = True  # only considers .gma-files

error_functions: List[EvalFunction] = [ReconstructionStats(sample_points=100000, cov_measure=True), ReconstructionStatsFiltered(ReconstructionStats(sample_points=100000, cov_measure=True))]

# Scaling options
scaling_active = False
scaling_interval = (0, 1)

# --- DO NOT CHANGE FROM HERE ---

#programs.execute_evaluation_singlepc_severalgm(fitpc_path, evalpc_path, gengmm_path,
programs.execute_evaluation_singlepc_severalgm(evalpc_path, None, gengmm_path,
                            error_functions, scaling_active, scaling_interval, gmaonly, modelpath=model_path)
