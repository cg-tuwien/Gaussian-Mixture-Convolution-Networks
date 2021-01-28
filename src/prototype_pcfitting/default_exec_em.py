from prototype_pcfitting import programs, RelChangeTerminationCriterion
from prototype_pcfitting.generators import EMGenerator
import datetime

# This takes a polygonal dataset, creates point clouds and then continues to generate gmms from it using classical EM.
# All the results are stored on disk.

# --- CONFIGUREABLE VARIABLES ---
# Define Paths
# Path to model .off-files
model_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/models"
# Path where to store the sampled pointclouds
genpc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/pointclouds"
# Path where to store the generated mixtures
# Are stored as .gma.ply-files (can be read in via gmc.mixture.read_gm_from_ply(path))
# And as .torch-files (can be read in with torch.load)
gengmm_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/gmms"

# Define Point Count (Samples), Gaussian Count and Batch Size (how many models to process at once)
n_points = 50000
n_gaussians = 512
batch_size = 1

# --- DO NOT CHANGE FROM HERE ---
# Define GMM Generators
generators = [
    EMGenerator(n_gaussians=n_gaussians, initialization_method='randnormpos', n_sample_points=n_points,
                termination_criterion=RelChangeTerminationCriterion(0.1, 20), em_step_points_subbatchsize=10000,
                em_step_gaussians_subbatchsize=512)
]
generator_identifiers = ["EM"]

# Read in Name
training_name = input('Name for this fitting (or empty for auto): ')
if training_name == '':
    training_name = f'fitPointcloud_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    print(training_name)

programs.execute_fitting(training_name=training_name,
                         n_points=n_points,
                         batch_size=batch_size,
                         generators=generators,
                         generator_identifiers=generator_identifiers,
                         model_path=model_path,
                         genpc_path=genpc_path,
                         gengmm_path=gengmm_path,
                         formats=[".gma.ply", ".torch"],
                         log_loss_console=20)
