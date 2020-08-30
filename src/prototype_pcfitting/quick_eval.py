from prototype_pcfitting import ErrorFunction, programs
from prototype_pcfitting.error_functions import LikelihoodLoss

# This takes a single GMM with a single point cloud and evluates it on a single error function
# The results are printed to the console.

# --- CONFIGUREABLE VARIABLES ---
# Define Paths
pc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/pointclouds/n20000/test/chair_0890.off"
gm_path = \
    "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/" + \
    "gmms/200830-03-corrected/GD/test/chair_0890.off.gma.ply"
gm_is_model = False

# pc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/pointclouds/n20000/train/chair_0030.off"
# gm_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/da-gm-1/da-gm-1/data/c_30HR.ply"
# gm_is_model = True

# Error Function
error_function: ErrorFunction = LikelihoodLoss()

# --- DO NOT CHANGE FROM HERE ---
programs.quick_evaluation(pc_path, gm_path, gm_is_model, error_function)