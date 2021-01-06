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
n_gaussians = 512
batch_size = 1

# Define GMM Generators
terminator = RelChangeTerminationCriterion(0.1, 100)
terminator2 = RelChangeTerminationCriterion(0.1, 20)

generators = [
    EckartGeneratorHP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, initialization_method='randnormpos', m_step_points_subbatchsize=10000, use_scaling=False),
    EckartGeneratorHP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, initialization_method='randnormpos', m_step_points_subbatchsize=10000, use_scaling=True, scaling_interval=(0, 1)),
    EckartGeneratorHP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, initialization_method='randnormpos', m_step_points_subbatchsize=10000, use_scaling=True, scaling_interval=(-50, 50))
# EckartGeneratorSP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, initialization_method='randnormpos', partition_treshold=0.0, m_step_points_subbatchsize=10000, e_step_pair_subbatchsize=5120000),
# EckartGeneratorSP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, initialization_method='randnormpos', partition_treshold=0.1, m_step_points_subbatchsize=10000, e_step_pair_subbatchsize=5120000),
# EckartGeneratorSP(n_gaussians_per_node=8, n_levels=3, termination_criterion=terminator2, initialization_method='randnormpos', partition_treshold=0.3, m_step_points_subbatchsize=10000, e_step_pair_subbatchsize=5120000),
]
generator_identifiers = ["EckHP-bed-8-3-extsc(1000)-intscF", "EckHP-bed-8-3-extsc(1000)-intsc(0-1)", "EckHP-bed-8-3-extsc(1000)-intsc(-50-50)"]


# Logging options (see readme.txt)
log_positions = 0
log_loss_console = 1
log_loss_tb = 1
log_rendering_tb = 5
log_gm = 1

# Scaling options
scaling_active = True
scaling_interval = (-1000, 1000)

# Read in Name
training_name = "DebugEck"

# pcbatch = DummyPcGenerator.generate_dummy_pc1()
# pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/realdummy1.off")
# pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/pointclouds/n1000000/test/chair_0895.off")
# pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/pointclouds/n20000/test/chair_0890.off")
# pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/pointclouds/n100000/test/chair_0890.off")
pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/pointclouds/n100000/bed_0003.off")
# pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/pointclouds/n10000/bed_0003.off")
# pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/pointclouds/n100000/bathtub_0001.off")
# pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/pointclouds/n1000/chair_0001.off")
# pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/daav/face01.off")
# pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_plane/pointclouds/n10000/plane0-original.off")
# pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_plane/pointclouds/n10000/plane3-rotated1.off")

# gmbatch = data_loading.read_gm_from_ply("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/EmMatlab/EmGm/output-n20000-g100-nofilter-highprec.gmm.ply", ismodel=True).double()
# gmbatch = data_loading.read_gm_from_ply("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/EmMatlab/EmGm/output.gmm.ply", ismodel=True)
gmbatch = None

programs.execute_fitting_on_single_pcbatch(training_name, pcbatch, gengmm_path, log_path, n_gaussians,
                                           generators, generator_identifiers, log_positions,
                                           log_loss_console, log_loss_tb, log_rendering_tb, log_gm,
                                           gmbatch, scaling_active, scaling_interval)

# programs.execute_fitting(training_name, model_path, genpc_path, gengmm_path, log_path, n_points, n_gaussians,
#                          batch_size, generators, generator_identifiers, log_positions, log_loss_console,
#                          log_loss_tb, log_rendering_tb, log_gm)