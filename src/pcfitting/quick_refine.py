from pcfitting import programs, MaxIterationTerminationCriterion
from pcfitting.generators import GradientDescentGenerator
import datetime

# This takes a single GMM and refines it.

# Define Paths
out_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/refinetest/"
pc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/pointclouds/n20000/train/chair_0030.off"
gm_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/da-gm-1/da-gm-1/data/c_30HR.ply"
gm_is_model = True

# pc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/pointclouds/n20000/test/chair_0890.off"
# gm_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/" + \
#           "gmms/200830-03-corrected/GD/test/chair_0890.off.gma.ply"
# gm_is_model = False

# Create Termination Criterions and GMM Generator
terminator = MaxIterationTerminationCriterion(1000)
generator = GradientDescentGenerator(n_gaussians=0, n_sample_points=1000, termination_criterion=terminator)

# Logging options (see readme.txt)
log_positions = 0
log_loss_console = 1
log_loss_tb = 1
log_rendering_tb = 250
log_gm = 250

# Scaling options
scaling_active = False
scaling_interval = (0, 1)


# --- DO NOT CHANGE FROM HERE ---
training_name = input('Name for this training (or empty for auto): ')
if training_name == '':
    training_name = f'quick_refine{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

programs.quick_refine(training_name, pc_path, gm_path, gm_is_model, out_path, generator,
                      log_positions, log_loss_console, log_loss_tb, log_rendering_tb, log_gm,
                      scaling_active, scaling_interval)