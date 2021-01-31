from prototype_pcfitting import programs, RelChangeTerminationCriterion
from prototype_pcfitting.generators import EckartGeneratorSP
import datetime

# This takes a polygonal dataset, creates point clouds and then continues to generate gmms from it using Eckart SP
# All the results are stored on disk

# --- CONFIGUREABLE VARIABLES ---
# Define Paths
# Path to model .off-files (or None)
model_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/models"
# Path where to store the sampled pointclouds (if model_path given) or where to read them from
pc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/pointclouds"
# Path where to store the generated mixtures
# Are stored as .gma.ply-files (can be read in via gmc.mixture.read_gm_from_ply(path))
# And as .torch-files (can be read in with torch.load)
gengmm_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/gmms"

# Define Point Count (Samples), Gaussian Count per Node, Levels
# Final Number of Gaussians is = Gaussian Count per Node ^ Levels
n_points = 50000
n_gaussians_per_node = 8
n_levels = 3

# --- DO NOT CHANGE FROM HERE ---
# Define GMM Generators
generators = [
    EckartGeneratorSP(n_gaussians_per_node=n_gaussians_per_node, n_levels=3, partition_threshold=0.1,
                      termination_criterion=RelChangeTerminationCriterion(0.1, 20),
                      initialization_method='bb', e_step_pair_subbatchsize=5120000, m_step_points_subbatchsize=10000)
]
generator_identifiers = ["EckSP"]

# Read in Name
training_name = input('Name for this fitting (or empty for auto): ')
if training_name == '':
    training_name = f'fitPointcloud_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

programs.execute_fitting(training_name=training_name,
                         n_points=n_points,
                         batch_size=1,
                         generators=generators,
                         generator_identifiers=generator_identifiers,
                         model_path=model_path,
                         genpc_path=pc_path,
                         gengmm_path=gengmm_path,
                         formats=[".gma.ply", ".torch"],
                         log_loss_console=20)
