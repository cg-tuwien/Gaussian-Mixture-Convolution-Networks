import typing

from pcfitting import programs, MaxIterationTerminationCriterion, RelChangeTerminationCriterion
from pcfitting.generators import EMGenerator, PreinerGenerator, EckartGeneratorSP
import pcfitting.modelnet_dataset_iterator
import pcfitting.config as general_config

# This takes a polygonal dataset, creates point clouds and then continues to generate gmms from it using classical EM.
# All the results are stored on disk.

# --- CONFIGUREABLE VARIABLES ---
# Define Paths
# Path to model .off-files (or None)
# model_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/models"
# Path where to store the sampled pointclouds (if model_path given) or where to read them from
pc_path = f"{general_config.data_base_path}/modelnet/pointclouds"


# Define Point Count (Samples), Gaussian Count and Batch Size (how many models to process at once)

class Config:
    def __init__(self, n_gaussians: int = 64, eps: float = 0.00005, gengmm_path: typing.Optional[str] = None):
        epsmul = 1000000
        assert eps * epsmul > 0
        self.n_points = 50000
        self.batch_size = 25
        self.n_gaussians = n_gaussians
        self.eps = eps
        self.name = f"fpsm_n{self.n_gaussians}_eps{int(self.eps * 1000000)}"

        # Path where to store the generated mixtures
        # Are stored as .gma.ply-files (can be read in via gmc.mixture.read_gm_from_ply(path))
        # And as .torch-files (can be read in with torch.load)
        self.gengmm_path = gengmm_path


def fit(config: Config):
    run(config.name,
        EMGenerator(n_gaussians=config.n_gaussians, initialization_method='fpsmax', n_sample_points=config.n_points,
                    termination_criterion=MaxIterationTerminationCriterion(0), em_step_points_subbatchsize=10000, eps=config.eps,
                    em_step_gaussians_subbatchsize=512, verbosity=general_config.verbosity),
        config.gengmm_path,
        config.batch_size
        )


def run(name, generator, gengmm_path, batch_size):
    generators = [generator, ]
    generator_identifiers = [name]

    log_loss = 0
    if general_config.verbosity > 2:
        log_loss = 20

    if gengmm_path is None:
        gengmm_path = f"{general_config.data_base_path}/modelnet/gmms"

    programs.execute_fitting2(training_name=None,
                              dataset=pcfitting.modelnet_dataset_iterator.ModelNetDatasetIterator(batch_size=batch_size, dataset_path=pc_path),
                              generators=generators,
                              generator_identifiers=generator_identifiers,
                              gengmm_path=gengmm_path,
                              formats=[".gma.ply", ".torch"],
                              log_loss_console=log_loss,
                              verbosity=1)
