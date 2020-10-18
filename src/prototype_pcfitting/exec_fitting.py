from prototype_pcfitting import programs, MaxIterationTerminationCriterion, RelChangeTerminationCriterion
from prototype_pcfitting.generators import GradientDescentGenerator, EMGenerator
import datetime

# This takes a polygonal dataset, creates point clouds and then continues to generate gmms from it
# All the results are stored on disk + Logs

# --- CONFIGUREABLE VARIABLES ---
# Define Paths (see readme.txt)
model_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/models"
genpc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/pointclouds"
gengmm_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/gmms"
log_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/logs"

# Define Point Count, Gaussian Count and Batch Size
n_points = 1000000
n_gaussians = 100
batch_size = 10  # ToDo: test with higher size

# Define GMM Generators
terminator1 = RelChangeTerminationCriterion(0.1, 100)
generators = [GradientDescentGenerator(n_gaussians=n_gaussians, n_sample_points=1000,
                                       termination_criterion=terminator1),
              EMGenerator(n_gaussians=n_gaussians, n_sample_points=20000,
                          termination_criterion=MaxIterationTerminationCriterion(101))]
generator_identifiers = ["GD", "EM"]

# Logging options (see readme.txt)
log_positions = 0
log_loss_console = 1
log_loss_tb = 1
log_rendering_tb = 50
log_gm = 50
log_seperate_directories = False

# --- DO NOT CHANGE FROM HERE ---
# Read in Name
training_name = input('Name for this training (or empty for auto): ')
if training_name == '':
    training_name = f'fitPointcloud_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

programs.execute_fitting(training_name, model_path, genpc_path, gengmm_path, log_path, n_points, n_gaussians,
                         batch_size, generators, generator_identifiers, log_positions, log_loss_console,
                         log_loss_tb, log_rendering_tb, log_gm, log_seperate_directories)
