# This is only for debugging purposes

from prototype_pcfitting import programs, MaxIterationTerminationCriterion, data_loading, RelChangeTerminationCriterion
from prototype_pcfitting.generators import EMGenerator, GradientDescentGenerator, EckartGenerator, EckartGenerator3


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
generators = [
    EckartGenerator3(n_gaussians_per_node=8, n_levels=3)
]
    # GradientDescentGenerator(n_gaussians=n_gaussians, n_sample_points=1000, initialization_method=0,
    #                                    termination_criterion=terminator),
    # GradientDescentGenerator(n_gaussians=n_gaussians, n_sample_points=1000, initialization_method=1,
    #                          termination_criterion=terminator),
    # GradientDescentGenerator(n_gaussians=n_gaussians, n_sample_points=1000, initialization_method=2,
    #                          termination_criterion=terminator),
    # GradientDescentGenerator(n_gaussians=n_gaussians, n_sample_points=1000, initialization_method=3,
    #                          termination_criterion=terminator)]
              #EMGenerator(n_gaussians=n_gaussians, n_sample_points=20000, initialization_method=0,
               #                       termination_criterion=MaxIterationTerminationCriterion(100))]
#generator_identifiers = ["EM"]
# generator_identifiers = ["GDi0", "GDi1", "GDi2", "GDi3"]
generator_identifiers = ["Eckart3"]

# Logging options (see readme.txt)
log_positions = 0
log_loss_console = 1
log_loss_tb = 1
log_rendering_tb = 10#5
log_gm = 10#1

# Read in Name
training_name = "Debug"

# pcbatch = DummyPcGenerator.generate_dummy_pc1()
# pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/realdummy1.off")
# pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/pointclouds/n1000000/test/chair_0895.off")
#pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/pointclouds/n20000/test/chair_0895.off")
pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/pointclouds/n100000/test/chair_0890.off")
# pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/pointclouds/n1000000/bathtub_0005.off")

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