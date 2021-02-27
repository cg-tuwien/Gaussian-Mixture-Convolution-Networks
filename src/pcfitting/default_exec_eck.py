from pcfitting import programs, RelChangeTerminationCriterion
from pcfitting.generators import EckartGeneratorSP
import pcfitting.modelnet_dataset_iterator
import datetime
import pcfitting.config as config

# This takes a polygonal dataset, creates point clouds and then continues to generate gmms from it using Eckart SP
# All the results are stored on disk

# --- CONFIGUREABLE VARIABLES ---
# Define Paths
# Path to model .off-files (or None)
model_path = f"{config.data_base_path}/modelnet/models"
# Path where to store the sampled pointclouds (if model_path given) or where to read them from
pc_path = f"{config.data_base_path}/modelnet/pointclouds"
# Path where to store the generated mixtures
# Are stored as .gma.ply-files (can be read in via gmc.mixture.read_gm_from_ply(path))
# And as .torch-files (can be read in with torch.load)
gengmm_path = f"{config.data_base_path}/modelnet/gmms"

# Define Point Count (Samples), Gaussian Count per Node, Levels
# Final Number of Gaussians is = Gaussian Count per Node ^ Levels
n_points = 50000
n_gaussians_per_node = 8
n_levels = 3

# --- DO NOT CHANGE FROM HERE ---
# Define GMM Generators
generators = [
    EckartGeneratorSP(n_gaussians_per_node=n_gaussians_per_node, n_levels=2, partition_threshold=0.1,
                      termination_criterion=RelChangeTerminationCriterion(0.1, 20),
                      initialization_method='bb', e_step_pair_subbatchsize=5120000, m_step_points_subbatchsize=10000)
]
generator_identifiers = ["EckSP64"]

# Read in Name
# training_name = input('Name for this fitting (or empty for auto): ')
# if training_name == '':
#     training_name = f'fitPointcloud_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

log_loss = 0
if config.verbosity > 2:
    log_loss = 20

programs.execute_fitting2(training_name=None,
                          dataset=pcfitting.modelnet_dataset_iterator.ModelNetDatasetIterator(batch_size=1, dataset_path=pc_path),
                          generators=generators,
                          generator_identifiers=generator_identifiers,
                          gengmm_path=gengmm_path,
                          formats=[".gma.ply", ".torch"],
                          log_loss_console=log_loss,
                          verbosity=config.verbosity)
