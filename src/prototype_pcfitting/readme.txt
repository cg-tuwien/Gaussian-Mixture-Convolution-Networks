THIS FILE IS NOT UP TO DATE.


This is an explanation on how to execute GMM fitting and evaluation.

GMM Fitting:
The file to execute is exec_fitting.py. This file takes polygonal model files, samples them to point clouds (unless
point clouds already exist), and generate GMMs from it using a specified list of GMM Generators.
At the beginning of execution the user has to enter a name for the training, which will be used to identify the
corresponding results in the directories.
In there there are a few variables that can be adapted to match one's needs:
model_path is the path to the .off-files, of the polygonal objects to sample the point clouds from.
    The .off files don't have to be in the directory itself, but can be in a subdirectory as well.
    All .off-files that start with the given path are considered.
genpc_path is the path, where the generated pointcloud .off-files will be stored. Also, pointclouds will be loaded
    from disk if there is already a file for this pointcloud in this directory. The pointclouds will not be stored
    directly in that given path, but in a subdirectory "nX", where X is the point count. In that subdirectory, the
    file will have the same path as the corresponding model file in model_path. So for example, if there is a polygonal
    model "<model_path>/train/chair_0890.off" then the corresponding pointcloud could be stored in "<genpc_path>/n100/
    test/chair_0890.off".
gengmm_path is the path, where the generated final GMMs will be stored as .ply-files. Per GMM, two files are stored:
    A .gma.ply-file, where the weights represent amplitudes, and a .gmm.ply-file, where the weights represent priors.
    The files will not be stored directly in the given path, but in subdirectories of the shape: "<gengmm_path>/
    <training_name>/<generator_identifier>/<model_name>.gma.ply" (or gmm.ply). model_name is the path of the model
    relative to model_path. generator_identifier is explained below. So for a model "<model_path>/train/chair_0890.off",
    the corresponding gmm file could be stored in "<gengmm_path>/200827-02-eval/GD/train/chair_0890.off.gma-ply".
log_path is a path where log files will be stored, such as tensorboard data, GMM-logs or position-logs (depending on
    logging options). The logging files will not be stored directly in that path, but in a subdirectory of shape
    "<log_path>/<training_name>/<generator_identifier>/<model_name>/", so a intermediate GMM might be stored in
    "<log_path>/200827-02-eval/GD/train/chair_0890.off/gmm-0250.gma.ply".
n_points defines how many points the sampled point clouds should have.
n_gaussians defines how many Gaussians the GMMs should have.
batch_size defines how many pointclouds should be processed as one
generators is a list of GMMGenerator objects, each represting a different GMM construction algorithm. Algorithm-
    specific options can be set here as well.
generator_identifiers is a list of strings, each being an identifier for a corrsponding generator from the generators
    list.
log_positions defines if Gaussian positions should be stored in binary files. If this is 0, no positions will be stored,
    if it is n > 0, then positions will be flushed to the binary files "pos-g<i>.bin", where i the number of the
    Gaussian, every nth iteration. These files can be opened with the visualizer to inspect Gaussian travel paths.
log_loss_console defines if the losses should be printed to the console. If this is 0, they won't be logged.
    If it is n > 0, then they will be logged every nth iteration.
log_loss_tb defines if the losses should be stored in tensorboard. Same principle as with log_loss_console.
log_loss_rendering_tb defines if visualizations should be stored in the tensorboard. Same principle as with the others.
    Per default, density and ellipsoid visualizations are stored.
log_gm defines if intermediate GMMs should be stored. Same principle as with the others.

GMM Evaluation:
The file to execute is exec_eval.py. This file reads in the generated GMMs and evaluates them on the corresponding
pointclouds with a given list of error functions. The results are printed to the console.
At the beginning of execution the user has to enter a name for the training to evaluate.
In there there are a few variables that can be adapted to match one's needs:
model_path Same as in GMM Fitting
genpc_path Same as in GMM Fitting
gengmm_path Same as in GMM Fitting
n_points Defines how many points the point clouds to load (or generate) should have
eval_points Defines how many points of the pointcloud should be used for evaluation. If this is smaller than n_points,
    a number of points will randomly be sampled from the pointcloud. If this is n_points, then the whole pointcloud
    will be used.
generator_identifiers Identifiers of generators that have been used in GMM fitting
error_functions List of error functions to use for evaluation
error_function_identifiers List of strings, each being an identifier for the corresponding error functions (for output).

Quick GMM Evaluation:
The file to execute is quick_eval.py. This reads in one GMM and one point cloud and evaluates them using a specified
error function.
The following variables can be specified:
pc_path Path to the pointcloud off file
gm_path Path to the GMM ply file
gm_is_model True if the weights in the file are priors (.gmm.ply), False if they are amplitudes (.gma.ply)
error_function Error function to use

Quick GMM Refinement:
The file to execute is quick_refine.ply. This reads in one GMM and refines it using a specified GMM generaotr.
The following variables can be specified:
out_path Path to store the result in. Logs are stored in a subdirectory "log".
pc_path Path to the pointcloud off file
gm_path Path to the GMM ply file
gm_is_model True if the weights in the file are priors (.gmm.ply), False if they are amplitudes (.gma.ply)
generator A GMMGenerator to use for refining
logging options: same as in GMM fitting

Possible ToDos:
 - Support Iterating over Point Clouds without models