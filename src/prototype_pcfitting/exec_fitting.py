from prototype_pcfitting import programs, MaxIterationTerminationCriterion, RelChangeTerminationCriterion
from prototype_pcfitting.generators import GradientDescentGenerator, EMGenerator, EckartGeneratorSP, EckartGeneratorHP
import datetime
import torch

# This takes a polygonal dataset, creates point clouds and then continues to generate gmms from it
# All the results are stored on disk + Logs

# --- CONFIGUREABLE VARIABLES ---
# Define Paths (see readme.txt)
# model_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/models"
# genpc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/pointclouds"
# gengmm_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/gmms"
# log_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/logs"
model_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_bunny/models"
genpc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_bunny/pointclouds"
gengmm_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_bunny/gmms"
log_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_bunny/logs"

# Define Point Count, Gaussian Count and Batch Size
n_points = 36000 #100000
n_gaussians = 512
batch_size = 1  # ToDo: test with higher size
continuing = True

# Define GMM Generators
terminator1 = RelChangeTerminationCriterion(0.1, 100)
terminator2 = RelChangeTerminationCriterion(0.1, 20)
generators = [
#     EMGenerator(n_gaussians=n_gaussians, initialization_method='randnormpos', n_sample_points=10000,
#                  termination_criterion=terminator2),
#     EMGenerator(n_gaussians=n_gaussians, initialization_method='fpsmax', n_sample_points=10000,
#                       termination_criterion=terminator2),
#     EckartGeneratorHP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, initialization_method='tight-bb', m_step_points_subbatchsize=10000),
#     EckartGeneratorHP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, initialization_method='randnormpos', m_step_points_subbatchsize=10000),
#     EckartGeneratorHP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, initialization_method='fpsmax', m_step_points_subbatchsize=10000),
#     EckartGeneratorHP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, initialization_method='kmeans', m_step_points_subbatchsize=10000),
#     EckartGeneratorHP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, initialization_method='eigen', m_step_points_subbatchsize=10000),
#     EckartGeneratorSP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, initialization_method='bb', partition_treshold=0.0, m_step_points_subbatchsize=10000, e_step_pair_subbatchsize=5120000),
#     EckartGeneratorSP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, initialization_method='randnormpos', partition_treshold=0.0, m_step_points_subbatchsize=10000, e_step_pair_subbatchsize=5120000),
#     EckartGeneratorSP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, initialization_method='fpsmax', partition_treshold=0.0, m_step_points_subbatchsize=10000, e_step_pair_subbatchsize=5120000),
#     EckartGeneratorSP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, initialization_method='kmeans', partition_treshold=0.0, m_step_points_subbatchsize=10000, e_step_pair_subbatchsize=5120000),
      EckartGeneratorSP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator1, initialization_method='bb', partition_treshold=0.1, m_step_points_subbatchsize=10000, e_step_pair_subbatchsize=5120000),
#     EckartGeneratorSP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, initialization_method='randnormpos', partition_treshold=0.1, m_step_points_subbatchsize=10000, e_step_pair_subbatchsize=5120000),
#     EckartGeneratorSP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, initialization_method='fpsmax', partition_treshold=0.1, m_step_points_subbatchsize=10000, e_step_pair_subbatchsize=5120000),
#     EckartGeneratorSP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, initialization_method='kmeans', partition_treshold=0.1, m_step_points_subbatchsize=10000, e_step_pair_subbatchsize=5120000),
#     EckartGeneratorSP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, initialization_method='bb', partition_treshold=0.3, m_step_points_subbatchsize=10000, e_step_pair_subbatchsize=5120000),
#     EckartGeneratorSP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, initialization_method='randnormpos', partition_treshold=0.3, m_step_points_subbatchsize=10000, e_step_pair_subbatchsize=5120000),
#     EckartGeneratorSP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, initialization_method='fpsmax', partition_treshold=0.3, m_step_points_subbatchsize=10000, e_step_pair_subbatchsize=5120000),
#     EckartGeneratorSP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, initialization_method='kmeans', partition_treshold=0.3, m_step_points_subbatchsize=10000, e_step_pair_subbatchsize=5120000),
]
generator_identifiers = ["EckSP0.1iBB"]
# generator_identifiers = ["00-EMiRNP", "01-EMiFspmax", "10-EckHPiTBB", "11-EckHPiRNP", "12-EckHPiFpsmax", "13-EckHPiKM", "14-EckHPiEigen",
#                          "20-EckSP0.0iBB","21-EckSP0.0iRNP", "22-EckSP0.0iFpsmax", "23-EckSP0.0iKM",
#                          "30-EckSP0.1iBB","31-EckSP0.1iRNP", "32-EckSP0.1iFpsmax", "33-EckSP0.1iKM",
#                          "40-EckSP0.3iBB","41-EckSP0.3iRNP", "42-EckSP0.3iFpsmax", "43-EckSP0.3iKM"]
#
# generators = [
#     EMGenerator(n_gaussians=n_gaussians, initialization_method="bb", n_sample_points=-1,
#                      termination_criterion=terminator2, eps=1e-7, dtype=torch.float32, use_noise_cluster=False),
#     EMGenerator(n_gaussians=n_gaussians, initialization_method="fpsmax", n_sample_points=-1,
#                      termination_criterion=terminator2, eps=1e-7, dtype=torch.float32, use_noise_cluster=False),
# ]
# generator_identifiers = ["EMrnp", "EMfpsmax"]

# Scaling options
scaling_active = False
scaling_interval = (0, 1)

# Logging options (see readme.txt)
log_positions = 0
log_loss_console = 1
log_loss_tb = 1
log_rendering_tb = 10
log_gm = 0 # 10
log_seperate_directories = True

# --- DO NOT CHANGE FROM HERE ---
# Read in Name
training_name = input('Name for this training (or empty for auto): ')
if training_name == '':
    training_name = f'fitPointcloud_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

programs.execute_fitting(training_name, model_path, genpc_path, gengmm_path, log_path, n_points, n_gaussians,
                         batch_size, generators, generator_identifiers, scaling_active, scaling_interval,
                         log_positions, log_loss_console, log_loss_tb, log_rendering_tb, log_gm,
                         log_seperate_directories, continuing)
