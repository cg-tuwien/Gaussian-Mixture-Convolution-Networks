from pcfitting import programs, MaxIterationTerminationCriterion, RelChangeTerminationCriterion
from pcfitting.generators import GradientDescentGenerator, EMGenerator, EckartGeneratorSP, EckartGeneratorHP, PreinerGenerator
import datetime
import torch

# This takes a polygonal dataset, creates point clouds and then continues to generate gmms from it
# All the results are stored on disk + Logs

# --- CONFIGUREABLE VARIABLES ---
# Define Paths (see readme.txt)
# model_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/models"
# genpc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vt_evaluation/fitpcs"
# gengmm_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vt_evaluation/gmms"
# log_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vt_evaluation/logs"
model_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_diff_scales/models"
genpc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_diff_scales/fitpcs"
gengmm_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_diff_scales/gmms"
log_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_diff_scales/logs"
# model_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_bunny/models"
# genpc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_bunny/pointclouds"
# gengmm_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_bunny/gmms"
# log_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_bunny/logs"

# Define Point Count, Gaussian Count and Batch Size
n_points = 100000
n_gaussians = 512
batch_size = 1
continuing = True

# Define GMM Generators
terminator1 = RelChangeTerminationCriterion(0.1, 100)
terminator2 = RelChangeTerminationCriterion(0.1, 20)
generators = [
      EMGenerator(n_gaussians=n_gaussians, initialization_method='fpsmax', termination_criterion=MaxIterationTerminationCriterion(0), em_step_points_subbatchsize=10000),
      EMGenerator(n_gaussians=n_gaussians, initialization_method='randnormpos', termination_criterion=terminator2, em_step_points_subbatchsize=10000),
      EMGenerator(n_gaussians=n_gaussians, initialization_method='fpsmax', termination_criterion=terminator2, em_step_points_subbatchsize=10000),
      EckartGeneratorSP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, initialization_method='bb', partition_threshold=0.1, m_step_points_subbatchsize=10000, e_step_pair_subbatchsize=5120000),
      EckartGeneratorSP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, initialization_method='randnormpos', partition_threshold=0.1, m_step_points_subbatchsize=10000, e_step_pair_subbatchsize=5120000),
      EckartGeneratorSP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, initialization_method='fpsmax', partition_threshold=0.1, m_step_points_subbatchsize=10000, e_step_pair_subbatchsize=5120000),
      EckartGeneratorSP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, initialization_method='eigen', partition_threshold=0.1, m_step_points_subbatchsize=10000, e_step_pair_subbatchsize=5120000),
      PreinerGenerator(fixeddist=0.9, ngaussians=512, alpha=5, avoidorphans=False)
]
generator_identifiers = ["fpsmax", "EMrnp", "EMfps", "Eckbb", "Eckrnp", "Eckfps", "Eckeigen", "Preiner-0.9-5"]
# generators = [
#       EMGenerator(n_gaussians=n_gaussians, initialization_method='fpsmax', termination_criterion=MaxIterationTerminationCriterion(0), em_step_points_subbatchsize=10000, eps=1e-4, eps_is_relative=False),
#       EMGenerator(n_gaussians=n_gaussians, initialization_method='fpsmax', termination_criterion=terminator2, em_step_points_subbatchsize=10000, eps=1e-4, eps_is_relative=False),
#       EckartGeneratorSP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, initialization_method='eigen', partition_threshold=0.1, m_step_points_subbatchsize=10000, e_step_pair_subbatchsize=5120000, eps=1e-4, eps_is_relative=False),
#       PreinerGenerator(fixeddist=0.9, ngaussians=512, alpha=5, avoidorphans=False)
# ]
# generator_identifiers = ["fpsmax", "EMfpsmax", "EckEigen", "Preiner"]

# Scaling options
scaling_active = False
scaling_interval = (0, 1)

# Logging options (see readme.txt)
log_positions = 0
log_loss_console = 1
log_loss_tb = 1
log_rendering_tb = 10
log_gm = 10

# --- DO NOT CHANGE FROM HERE ---
# Read in Name
training_name = input('Name for this training (or empty for auto): ')
if training_name == '':
    training_name = f'fitPointcloud_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

programs.execute_fitting(training_name=training_name,
                         n_points=n_points,
                         batch_size=batch_size,
                         generators=generators,
                         generator_identifiers=generator_identifiers,
                         model_path=model_path,
                         genpc_path=genpc_path,
                         gengmm_path=gengmm_path,
                         formats=[".gma.ply", ".torch"],
                         log_path=log_path,
                         scaling_active=scaling_active,
                         scaling_interval=scaling_interval,
                         log_positions=log_positions,
                         log_loss_console=log_loss_console,
                         log_loss_tb=log_loss_tb,
                         log_rendering_tb=log_rendering_tb,
                         log_gm=log_gm,
                         log_n_gaussians=n_gaussians,
                         continuing=continuing)
