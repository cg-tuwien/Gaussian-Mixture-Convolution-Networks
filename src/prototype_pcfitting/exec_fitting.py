from prototype_pcfitting import programs, MaxIterationTerminationCriterion, RelChangeTerminationCriterion
from prototype_pcfitting.generators import GradientDescentGenerator, EMGenerator, EckartGeneratorSP, EckartGeneratorHP
import datetime
import torch

# This takes a polygonal dataset, creates point clouds and then continues to generate gmms from it
# All the results are stored on disk + Logs

# --- CONFIGUREABLE VARIABLES ---
# Define Paths (see readme.txt)
model_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/models"
genpc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/pointclouds"
gengmm_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/gmms"
log_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/logs"

# Define Point Count, Gaussian Count and Batch Size
n_points = 100000
n_gaussians = 512
batch_size = 1  # ToDo: test with higher size

# Define GMM Generators
terminator1 = RelChangeTerminationCriterion(0.1, 100)
terminator2 = RelChangeTerminationCriterion(0.1, 20)
generators = [
    EMGenerator(n_gaussians=n_gaussians, initialization_method=0, n_sample_points=10000,
                 termination_criterion=terminator2),
    EMGenerator(n_gaussians=n_gaussians, initialization_method=3, n_sample_points=10000,
                      termination_criterion=terminator2),
    EckartGeneratorHP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, m_step_points_subbatchsize=10000),
    EckartGeneratorSP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, partition_treshold=0.3, m_step_points_subbatchsize=10000),
    EckartGeneratorSP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, partition_treshold=0.5, m_step_points_subbatchsize=10000)
]
generator_identifiers = ["EMi0", "EMi3", "EckHP", "EckSP0.3", "EckSP0.5"]
# generators = [
#     EMGenerator(n_gaussians=n_gaussians, initialization_method=0, m_step_points_subbatchsize=10000,
#                 termination_criterion=terminator2),
#     EMGenerator(n_gaussians=n_gaussians, initialization_method=0, n_sample_points=10000,
#                      termination_criterion=terminator2),
# ]
# generator_identifiers = ["EMFull", "EMSampled"]

# Logging options (see readme.txt)
log_positions = 0
log_loss_console = 1
log_loss_tb = 1
log_rendering_tb = 10
log_gm = 10
log_seperate_directories = True

# --- DO NOT CHANGE FROM HERE ---
# Read in Name
training_name = input('Name for this training (or empty for auto): ')
if training_name == '':
    training_name = f'fitPointcloud_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

programs.execute_fitting(training_name, model_path, genpc_path, gengmm_path, log_path, n_points, n_gaussians,
                         batch_size, generators, generator_identifiers, log_positions, log_loss_console,
                         log_loss_tb, log_rendering_tb, log_gm, log_seperate_directories)
