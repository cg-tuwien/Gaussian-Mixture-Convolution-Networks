# This is only for debugging purposes

from pcfitting import programs, MaxIterationTerminationCriterion, data_loading, RelChangeTerminationCriterion
from pcfitting.generators import EMGenerator, GradientDescentGenerator, EckartGeneratorHP, EckartGeneratorSP, PreinerGenerator
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
terminator1 = RelChangeTerminationCriterion(0.1, 100)
terminator2 = RelChangeTerminationCriterion(0.1, 50)

generators = [
    # PreinerGenerator(fixeddist=0.5, stdev=0.01, reductionFactor=1.25, levels=30, alpha=4.0),
    # PreinerGenerator(fixeddist=0.25, stdev=0.01, ngaussians=512, alpha=5.0),
    # PreinerGenerator(fixeddist=0.25, stdev=0.01, ngaussians=512, alpha=10.0),
    PreinerGenerator(fixeddist=0.5, stdev=0.01, ngaussians=512, alpha=1000.0),
    # PreinerGenerator(fixeddist=0.5, stdev=0.01, reductionFactor=1.25, levels=30, alpha=2.0),
    # PreinerGenerator(fixeddist=0.5, stdev=0.01, reductionFactor=3.0, levels=30, alpha=2.0),
]
generator_identifiers = [
    # "Preiner-fd0.25-stdev=0.01-a=05-ng=512",
    # "Preiner-fd0.25-stdev=0.01-a=10-ng=512",
    "Preiner-fd0.5-stdev=0.01-a=1000-ng=512",
]


# Logging options (see readme.txt)
log_positions = 0
log_loss_console = 1
log_loss_tb = 1
log_rendering_tb = 5
log_gm = 1

# Scaling options
scaling_active = False
scaling_interval = (0, 1)

# Read in Name
training_name = "DebugPreiner"

# pcbatch = DummyPcGenerator.generate_dummy_pc1()
# pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/realdummy1.off")
# pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/pointclouds/n1000000/test/chair_0895.off")
# pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/pointclouds/n20000/test/chair_0890.off")
# pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/pointclouds/n100000/test/chair_0890.off")
pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/pointclouds/n100000/bed_0003.off")
# pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/pointclouds/n100000/bed_0003_scaled.off")
# pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/pointclouds/n1000/chair_0001.off")
# pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/stanford-bunny/bunny/reconstruction/bun_zipper_pc.off")

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