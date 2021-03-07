from pcfitting import EvalFunction, programs
from pcfitting.error_functions import LikelihoodLoss, ReconstructionStats

# This takes a single GMM with a single point cloud and evluates it on a single error function
# The results are printed to the console.

# --- CONFIGUREABLE VARIABLES ---
# Define Paths
pc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_diff_scales/fitpcs/n100000/bathtub-0001-3-double.off"
gm_path = \
    r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\data\dataset_diff_scales\gmms\210306-01-EmEckPre\Eckeigen\bathtub-0001-3-double.off.gma.ply"
mesh_path = r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\data\dataset_diff_scales\models\bathtub-0001-3-double.off"
gm_is_model = False

# pc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/realdummy1.off"
# gm_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/logs/Debug/EM/1/gmm-00000.gma.ply"
# gm_is_model = False

# Error Function
error_function: EvalFunction = ReconstructionStats()

# Scaling options
scaling_active = False
scaling_interval = (0, 1)

# --- DO NOT CHANGE FROM HERE ---
programs.quick_evaluation(pc_path, gm_path, mesh_path, gm_is_model, error_function, scaling_active, scaling_interval)