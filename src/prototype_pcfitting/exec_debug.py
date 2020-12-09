# This is only for debugging purposes

from prototype_pcfitting import programs, MaxIterationTerminationCriterion, data_loading, RelChangeTerminationCriterion
from prototype_pcfitting.generators import EMGenerator, GradientDescentGenerator, EckartGeneratorHP, EckartGeneratorSP
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
n_gaussians = 100
batch_size = 1

# Define GMM Generators
terminator = RelChangeTerminationCriterion(0.1, 100)
terminator2 = RelChangeTerminationCriterion(0.1, 50)
# generators = [
#     # EMGenerator(n_gaussians=512, n_sample_points=100000, termination_criterion=terminator2, dtype=torch.float64,
#     #            m_step_points_subbatchsize=10000, initialization_method=3)
#     # EckartGeneratorSP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2,
#     #                   partition_treshold=0.0)#, m_step_gaussians_subbatchsize=128, m_step_points_subbatchsize=50000),
#     EMGenerator(n_gaussians=512, initialization_method=0, n_sample_points=10000,
#                 termination_criterion=terminator2),
# ]
# generator_identifiers = ["tEMx-s"]
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
#     EMGenerator(n_gaussians=512, n_sample_points=10000, termination_criterion=terminator2, dtype=torch.float64),
#     EckartGeneratorHP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2),
#     EckartGeneratorSP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2,
#                       partition_treshold=0.0, m_step_gaussians_subbatchsize=128, m_step_points_subbatchsize=50000),
#     EckartGeneratorSP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2,
#                       partition_treshold=0.1, m_step_gaussians_subbatchsize=128, m_step_points_subbatchsize=50000),
#     EckartGeneratorSP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2,
#                       partition_treshold=0.3, m_step_gaussians_subbatchsize=128, m_step_points_subbatchsize=50000),
#     EckartGeneratorSP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2,
#                       partition_treshold=0.5, m_step_gaussians_subbatchsize=128, m_step_points_subbatchsize=50000),
#     EckartGeneratorSP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2,
#                       partition_treshold=0.8, m_step_gaussians_subbatchsize=128, m_step_points_subbatchsize=50000),
# ]
# generator_identifiers = ["EM", "EckHP", "EckSP-l=0.0", "EckSP-l=0.1", "EckSP-l=0.3", "EckSP-l=0.5", "EckSP-l=0.8"]

# Logging options (see readme.txt)
log_positions = 0
log_loss_console = 1
log_loss_tb = 1
log_rendering_tb = 5
log_gm = 1

# Read in Name
training_name = "DebugX"

# pcbatch = DummyPcGenerator.generate_dummy_pc1()
# pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/realdummy1.off")
# pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/pointclouds/n1000000/test/chair_0895.off")
# pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/pointclouds/n20000/test/chair_0890.off")
pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/pointclouds/n100000/test/chair_0890.off")
#pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/pointclouds/n100000/bathtub_0001.off")

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