from prototype_pcfitting import ErrorFunction, programs
from prototype_pcfitting.error_functions import LikelihoodLoss

# This takes a single GMM with a single point cloud and evluates it on a single error function
# The results are printed to the console.

# --- CONFIGUREABLE VARIABLES ---
# Define Paths
pc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/pointclouds/n100000/bathtub_0001.off"
gm_path = \
    "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/logs/DebugEck/1609349674-XSP/1/gmm-00000.gma.ply"
gm_is_model = False

# pc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/realdummy1.off"
# gm_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/logs/Debug/EM/1/gmm-00000.gma.ply"
# gm_is_model = False

# Error Function
error_function: ErrorFunction = LikelihoodLoss()

# Scaling options
scaling_active = False
scaling_interval = (0, 1)

# --- DO NOT CHANGE FROM HERE ---
programs.quick_evaluation(pc_path, gm_path, gm_is_model, error_function, scaling_active, scaling_interval)