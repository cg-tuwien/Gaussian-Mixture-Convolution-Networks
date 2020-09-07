# This is only for debugging purposes

from prototype_pcfitting import programs, MaxIterationTerminationCriterion
from prototype_pcfitting.generators import EMGenerator


# Define Paths (see readme.txt)
model_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/models"
genpc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/pointclouds"
gengmm_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/gmms"
log_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/logs"

# Define Point Count, Gaussian Count and Batch Size
n_points = 20000
n_gaussians = 100
batch_size = 1

# Define GMM Generators
# terminator = RelChangeTerminationCriterion(0.1, 250)
generators = [EMGenerator(n_gaussians=n_gaussians, n_sample_points=1000,
                                       termination_criterion=MaxIterationTerminationCriterion(1000))]
generator_identifiers = ["EM"]

# Logging options (see readme.txt)
log_positions = 0
log_loss_console = 1
log_loss_tb = 0
log_rendering_tb = 0
log_gm = 1

# Read in Name
training_name = "Debug"

programs.execute_fitting(training_name, model_path, genpc_path, gengmm_path, log_path, n_points, n_gaussians,
                         batch_size, generators, generator_identifiers, log_positions, log_loss_console,
                         log_loss_tb, log_rendering_tb, log_gm)