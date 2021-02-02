from prototype_pcfitting import programs, RelChangeTerminationCriterion
from prototype_pcfitting.generators import EMGenerator
import prototype_pcfitting.modelnet_dataset_iterator
import datetime
import prototype_pcfitting.config as config

# This takes a polygonal dataset, creates point clouds and then continues to generate gmms from it using classical EM.
# All the results are stored on disk.

# --- CONFIGUREABLE VARIABLES ---
# Define Paths
# Path to model .off-files (or None)
# model_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/models"
# Path where to store the sampled pointclouds (if model_path given) or where to read them from
pc_path = f"{config.data_base_path}/modelnet/pointclouds"
# Path where to store the generated mixtures
# Are stored as .gma.ply-files (can be read in via gmc.mixture.read_gm_from_ply(path))
# And as .torch-files (can be read in with torch.load)
gengmm_path = f"{config.data_base_path}/modelnet/gmms"

# Define Point Count (Samples), Gaussian Count and Batch Size (how many models to process at once)
n_points = 50000
batch_size = 100

# --- DO NOT CHANGE FROM HERE ---
# Define GMM Generators
generators = [
    EMGenerator(n_gaussians=32, initialization_method='randnormpos', n_sample_points=n_points,
                termination_criterion=RelChangeTerminationCriterion(0.1, 20), em_step_points_subbatchsize=10000,
                em_step_gaussians_subbatchsize=512, verbosity=config.verbosity),
    # EMGenerator(n_gaussians=64, initialization_method='randnormpos', n_sample_points=n_points,
    #             termination_criterion=RelChangeTerminationCriterion(0.1, 20), em_step_points_subbatchsize=10000,
    #             em_step_gaussians_subbatchsize=512, verbosity=config.verbosity),
    # EMGenerator(n_gaussians=128, initialization_method='randnormpos', n_sample_points=n_points,
    #             termination_criterion=RelChangeTerminationCriterion(0.1, 20), em_step_points_subbatchsize=10000,
    #             em_step_gaussians_subbatchsize=512, verbosity=config.verbosity),
    # EMGenerator(n_gaussians=256, initialization_method='randnormpos', n_sample_points=n_points,
    #             termination_criterion=RelChangeTerminationCriterion(0.1, 20), em_step_points_subbatchsize=10000,
    #             em_step_gaussians_subbatchsize=512, verbosity=config.verbosity),
    # EMGenerator(n_gaussians=512, initialization_method='randnormpos', n_sample_points=n_points,
    #             termination_criterion=RelChangeTerminationCriterion(0.1, 20), em_step_points_subbatchsize=10000,
    #             em_step_gaussians_subbatchsize=512, verbosity=config.verbosity),
]
generator_identifiers = ["EM32"]  # , "EM64", "EM128", "EM256", "EM512"]

programs.execute_fitting2(training_name=None,
                          dataset=prototype_pcfitting.modelnet_dataset_iterator.ModelNetDatasetIterator(batch_size=batch_size, dataset_path=pc_path),
                          generators=generators,
                          generator_identifiers=generator_identifiers,
                          gengmm_path=gengmm_path,
                          formats=[".gma.ply", ".torch"],
                          log_loss_console=0,
                          verbosity=config.verbosity)
