# This is only for debugging purposes

from prototype_pcfitting import programs, MaxIterationTerminationCriterion, data_loading, RelChangeTerminationCriterion
from prototype_pcfitting.generators import EMGenerator, GradientDescentGenerator, EckartGeneratorHP, EckartGeneratorSP, PreinerGenerator
import torch

# Define Paths (see readme.txt)
model_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/models"
genpc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/pointclouds"
gengmm_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/gmms"
log_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/logs"

# Define Point Count, Gaussian Count and Batch Size
# n_points = 20000
# n_points = 50
# n_gaussians = 100
# n_gaussians = 10000
n_gaussians = 256
batch_size = 1

# Define GMM Generators
terminator = RelChangeTerminationCriterion(0.1, 100)
terminator2 = RelChangeTerminationCriterion(0.1, 50)

generators = [
    EckartGeneratorHP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, m_step_points_subbatchsize=10000)
    #EckartGeneratorHP(n_gaussians_per_node=2, n_levels=7, termination_criterion=terminator2, m_step_points_subbatchsize=10000)
    #EckartGeneratorSP(n_gaussians_per_node=4, n_levels=4, termination_criterion=terminator2, partition_treshold=0.5, m_step_points_subbatchsize=10000)
]
generator_identifiers = ["EckHP-loss3"]

# generators = [
#     EckartGeneratorHP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, m_step_points_subbatchsize=10000),
#     EckartGeneratorSP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, partition_treshold=0.3, m_step_points_subbatchsize=10000),
#     EckartGeneratorSP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, partition_treshold=0.5, m_step_points_subbatchsize=10000),
#     EckartGeneratorHP(n_gaussians_per_node=4, n_levels=4, termination_criterion=terminator2, m_step_points_subbatchsize=10000),
#     EckartGeneratorSP(n_gaussians_per_node=4, n_levels=4, termination_criterion=terminator2, partition_treshold=0.3, m_step_points_subbatchsize=10000),
#     EckartGeneratorSP(n_gaussians_per_node=4, n_levels=4, termination_criterion=terminator2, partition_treshold=0.5, m_step_points_subbatchsize=10000),
#     EckartGeneratorHP(n_gaussians_per_node=2, n_levels=7, termination_criterion=terminator2, m_step_points_subbatchsize=10000),
#     EckartGeneratorSP(n_gaussians_per_node=2, n_levels=7, termination_criterion=terminator2, partition_treshold=0.3, m_step_points_subbatchsize=10000),
#     EckartGeneratorSP(n_gaussians_per_node=2, n_levels=7, termination_criterion=terminator2, partition_treshold=0.5, m_step_points_subbatchsize=10000),
# ]
# generator_identifiers = ["EckHP-8-3", "EckSP0.3-8-3", "EckSP0.5-8-3",
#                          "EckHP-4-4", "EckSP0.3-4-4", "EckSP0.5-4-4",
#                          "EckHP-2-7", "EckSP0.3-2-7", "EckSP0.5-2-7"]

# Logging options (see readme.txt)
log_positions = 0
log_loss_console = 1
log_loss_tb = 1
log_rendering_tb = 5
log_gm = 1

# Read in Name
training_name = "DebugEck"

# pcbatch = DummyPcGenerator.generate_dummy_pc1()
# pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/realdummy1.off")
# pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/pointclouds/n1000000/test/chair_0895.off")
# pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/pointclouds/n20000/test/chair_0890.off")
# pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/pointclouds/n100000/test/chair_0890.off")
pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/pointclouds/n100000/bed_0003.off")
#pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/pointclouds/n1000/chair_0001.off")

# gmbatch = data_loading.read_gm_from_ply("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/EmMatlab/EmGm/output-n20000-g100-nofilter-highprec.gmm.ply", ismodel=True).double().cuda()
# gmbatch = data_loading.read_gm_from_ply("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/EmMatlab/EmGm/output.gmm.ply", ismodel=True).cuda()
gmbatch = None

programs.execute_fitting_on_single_pcbatch(training_name, pcbatch, gengmm_path, log_path, n_gaussians,
                                                           generators, generator_identifiers, log_positions,
                                                           log_loss_console, log_loss_tb, log_rendering_tb, log_gm,
                                                            gmbatch)

# programs.execute_fitting(training_name, model_path, genpc_path, gengmm_path, log_path, n_points, n_gaussians,
#                          batch_size, generators, generator_identifiers, log_positions, log_loss_console,
#                          log_loss_tb, log_rendering_tb, log_gm)