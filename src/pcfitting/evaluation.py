import os

from pcfitting.eval_db_access import EvalDbAccess
from pcfitting import programs, MaxIterationTerminationCriterion, RelChangeTerminationCriterion, PCDatasetIterator, \
    data_loading, GMSampler
from pcfitting.generators import GradientDescentGenerator, EMGenerator, EckartGeneratorSP, EckartGeneratorHP, PreinerGenerator
from pcfitting.error_functions import AvgDensities, ReconstructionStats, GMMStats
import time
import gmc.mixture
from gmc.cpp.gm_vis.gm_vis import GMVisualizer
import matplotlib
import matplotlib.image as mimg
matplotlib.use('TkAgg')

model_path = r"F:\DA-Eval\dataset\models20"
fitpc_path = r"F:\DA-Eval\dataset\fitpcs"
evalpc_path = r"F:\DA-Eval\dataset\evalpcs"
recpc_path = r"F:\DA-Eval\dataset\recpcs"
gengmm_path = r"F:\DA-Eval\dataset\gmms"
rendering_path = r"F:\DA-Eval\dataset\renderings"

initterm = MaxIterationTerminationCriterion(0)
terminator2 = RelChangeTerminationCriterion(0.1, 20)

# -- EXECUTED --
# EMGenerator(n_gaussians=64, initialization_method='randnormpos', termination_criterion=terminator2, em_step_points_subbatchsize=10000, verbosity=0),
# EMGenerator(n_gaussians=64, initialization_method='randresp', termination_criterion=terminator2, em_step_points_subbatchsize=10000, verbosity=0),
# EMGenerator(n_gaussians=64, initialization_method='fpsmax', termination_criterion=terminator2, em_step_points_subbatchsize=10000, verbosity=0),
# EMGenerator(n_gaussians=64, initialization_method='kmeans', termination_criterion=terminator2, em_step_points_subbatchsize=10000, verbosity=0),
# EMGenerator(n_gaussians=256, initialization_method='randnormpos', termination_criterion=terminator2, em_step_points_subbatchsize=10000, verbosity=0),
# EMGenerator(n_gaussians=256, initialization_method='randresp', termination_criterion=terminator2, em_step_points_subbatchsize=10000, verbosity=0),
# EMGenerator(n_gaussians=256, initialization_method='fpsmax', termination_criterion=terminator2, em_step_points_subbatchsize=10000, verbosity=0),
# EMGenerator(n_gaussians=256, initialization_method='kmeans', termination_criterion=terminator2, em_step_points_subbatchsize=10000, verbosity=0),
# EMGenerator(n_gaussians=512, initialization_method='randnormpos', termination_criterion=terminator2, em_step_points_subbatchsize=10000, verbosity=0),
# EMGenerator(n_gaussians=512, initialization_method='randresp', termination_criterion=terminator2, em_step_points_subbatchsize=10000, verbosity=0),
# EMGenerator(n_gaussians=512, initialization_method='fpsmax', termination_criterion=terminator2, em_step_points_subbatchsize=10000, verbosity=0),
# EMGenerator(n_gaussians=512, initialization_method='kmeans', termination_criterion=terminator2, em_step_points_subbatchsize=10000, verbosity=0),

# EMGenerator(n_gaussians=64, initialization_method='fpsmax', termination_criterion=initterm, em_step_points_subbatchsize=10000, verbosity=0),
# EMGenerator(n_gaussians=256, initialization_method='fpsmax', termination_criterion=initterm, em_step_points_subbatchsize=10000, verbosity=0),
# EMGenerator(n_gaussians=512, initialization_method='fpsmax', termination_criterion=initterm, em_step_points_subbatchsize=10000, verbosity=0),

# -- IN PROGRESS --

# EMGenerator(n_gaussians=512, initialization_method='fpsmax', termination_criterion=initterm, em_step_points_subbatchsize=10000, verbosity=0, eps=1e-5),
# EMGenerator(n_gaussians=512, initialization_method='randnormpos', termination_criterion=terminator2, em_step_points_subbatchsize=10000, verbosity=0, eps=1e-5),
# EMGenerator(n_gaussians=512, initialization_method='randresp', termination_criterion=terminator2, em_step_points_subbatchsize=10000, verbosity=0, eps=1e-5),
# EMGenerator(n_gaussians=512, initialization_method='fpsmax', termination_criterion=terminator2, em_step_points_subbatchsize=10000, verbosity=0, eps=1e-5),
# EMGenerator(n_gaussians=512, initialization_method='kmeans', termination_criterion=terminator2, em_step_points_subbatchsize=10000, verbosity=0, eps=1e-5),

generators = [
    # EMGenerator(n_gaussians=64, initialization_method='fpsmax', termination_criterion=initterm, em_step_points_subbatchsize=10000, verbosity=0),
    # EMGenerator(n_gaussians=256, initialization_method='fpsmax', termination_criterion=initterm, em_step_points_subbatchsize=10000, verbosity=0),
    # EMGenerator(n_gaussians=512, initialization_method='fpsmax', termination_criterion=initterm, em_step_points_subbatchsize=10000, verbosity=0, eps=1e-5),
    EMGenerator(n_gaussians=64, initialization_method='randnormpos', termination_criterion=terminator2, em_step_points_subbatchsize=10000, verbosity=0),
    # EMGenerator(n_gaussians=64, initialization_method='randresp', termination_criterion=terminator2, em_step_points_subbatchsize=10000, verbosity=0),
    # EMGenerator(n_gaussians=64, initialization_method='fpsmax', termination_criterion=terminator2, em_step_points_subbatchsize=10000, verbosity=0),
    # EMGenerator(n_gaussians=64, initialization_method='kmeans', termination_criterion=terminator2, em_step_points_subbatchsize=10000, verbosity=0),
    # EMGenerator(n_gaussians=256, initialization_method='randnormpos', termination_criterion=terminator2, em_step_points_subbatchsize=10000, verbosity=0),
    # EMGenerator(n_gaussians=256, initialization_method='randresp', termination_criterion=terminator2, em_step_points_subbatchsize=10000, verbosity=0),
    # EMGenerator(n_gaussians=256, initialization_method='fpsmax', termination_criterion=terminator2, em_step_points_subbatchsize=10000, verbosity=0),
    # EMGenerator(n_gaussians=256, initialization_method='kmeans', termination_criterion=terminator2, em_step_points_subbatchsize=10000, verbosity=0),
    # EMGenerator(n_gaussians=512, initialization_method='randnormpos', termination_criterion=terminator2, em_step_points_subbatchsize=10000, verbosity=0, eps=1e-5),
    # EMGenerator(n_gaussians=512, initialization_method='randresp', termination_criterion=terminator2, em_step_points_subbatchsize=10000, verbosity=0, eps=1e-5),
    # EMGenerator(n_gaussians=512, initialization_method='fpsmax', termination_criterion=terminator2, em_step_points_subbatchsize=10000, verbosity=0, eps=1e-5),
    # EMGenerator(n_gaussians=512, initialization_method='kmeans', termination_criterion=terminator2, em_step_points_subbatchsize=10000, verbosity=0, eps=1e-5),
]

n_fit_points = 100000
n_eval_points_density = 1000000
n_eval_points_distance = 100000

dataset_fit = PCDatasetIterator(1, n_fit_points, fitpc_path, model_path)
dataset_eval_dens = PCDatasetIterator(1, n_eval_points_density, evalpc_path, model_path)
dataset_eval_dist = PCDatasetIterator(1, n_eval_points_distance, evalpc_path, model_path)
dbaccess = EvalDbAccess(r"F:\DA-Eval\Eval01.db")
evaldensity = AvgDensities()
evaldistane = ReconstructionStats()
evalstats = GMMStats()
vis = GMVisualizer(False, 800, 800)
vis.set_camera_auto(True)
vis.set_density_rendering(True)
vis.set_ellipsoids_pc_rendering(False, True, False)

batchcount = dataset_fit.remaining_batches_count()

i = -1
while dataset_fit.has_next():
    i += 1
    batch, names = dataset_fit.next_batch()
    batch_eval_dens, _ = dataset_eval_dens.next_batch()
    batch_eval_dist, _ = dataset_eval_dist.next_batch()
    for j in range(len(generators)):
        print(100 * ((i / batchcount) + (j / len(generators) / batchcount)), "%")
        print("Generator ", (j+1), "/", len(generators), " on ", names)

        termcrit = "RelChange(0.1,20)"#"MaxIter(0)"

        if dbaccess.has_em_run(names[0], n_fit_points, generators[j]._n_gaussians, n_fit_points, termcrit,
                               generators[j]._initialization_method, "float32", 1e-7, True):
            print("Skip")
            continue

        modelpath = os.path.join(model_path, names[0])

        print ("Fitting")

        start = time.time()

        gmbatch, gmmbatch = generators[j].generate(batch)

        end = time.time()

        # Evaluate
        print ("Evaluating")
        densvalues_fit = evaldensity.calculate_score_packed(batch, gmbatch, modelpath=modelpath)
        densvalues_eval = evaldensity.calculate_score_packed(batch_eval_dens, gmbatch, modelpath=modelpath)
        statvalues = evalstats.calculate_score_packed(batch, gmbatch, modelpath=modelpath)
        reconstructed = GMSampler.sample(gmmbatch, n_eval_points_distance)
        distvalues = evaldistane.calculate_score_on_reconstructed(batch_eval_dist, reconstructed, modelpath=modelpath)

        # Render
        print ("Rendering")
        vis.set_pointclouds(reconstructed.cpu())
        vis.set_gaussian_mixtures(gmbatch.cpu())
        res = vis.render()

        # Save in DB
        print ("Saving")
        emid = dbaccess.insert_options_em(n_fit_points, termcrit, generators[j]._initialization_method,
                                          "float32", 1e-7, True)
        runid = dbaccess.insert_run(names[0], n_fit_points, generators[j]._n_gaussians, gmbatch.shape[2],
                                    "EM", emid, (end - start))
        # TODO: FIX VALUES
        dbaccess.insert_density_eval(densvalues_fit[0].item(), densvalues_fit[1].item(), densvalues_fit[2].item(),
                                     densvalues_fit[4].item(), densvalues_fit[3].item(), densvalues_fit[5].item(),
                                     densvalues_fit[6].item(), None, runid, True, n_eval_points_density)
        dbaccess.insert_density_eval(densvalues_eval[0].item(), densvalues_eval[2].item(), densvalues_eval[3].item(),
                                     densvalues_eval[6].item(), densvalues_eval[4].item(), densvalues_eval[8].item(),
                                     densvalues_eval[9].item(), None, runid, False, n_eval_points_density)
        dbaccess.insert_distance_eval(distvalues[0].item(), distvalues[2].item(), distvalues[4].item(),
                                      distvalues[6].item(), distvalues[8].item(), distvalues[10].item(),
                                      distvalues[12].item(), distvalues[14].item(), distvalues[16].item(),
                                      distvalues[1].item(), distvalues[3].item(), distvalues[5].item(),
                                      distvalues[7].item(), distvalues[9].item(), distvalues[11].item(),
                                      distvalues[13].item(), distvalues[15].item(), distvalues[17].item(),
                                      distvalues[18].item(), distvalues[20].item(), distvalues[22].item(),
                                      distvalues[23].item(), runid, False, n_eval_points_distance)
        dbaccess.insert_stat_eval(statvalues[0].item(), statvalues[1].item(), statvalues[2].item(),
                                  statvalues[3].item(), statvalues[4].item(), statvalues[5].item(),
                                  statvalues[6].item(), statvalues[7].item(), statvalues[8].item(),
                                  statvalues[9].item(), statvalues[10].item(), statvalues[11].item(),
                                  statvalues[12].item(), statvalues[13].item(), statvalues[14].item(),
                                  statvalues[15].item(), statvalues[16].item(), statvalues[17].item(), runid)

        # mimg.imsave(os.path.join(rendering_path, "recpc-" + str(runid).zfill(9) + ".png"), res[0, 0])
        # mimg.imsave(os.path.join(rendering_path, "density-" + str(runid).zfill(9) + ".png"), res[0, 1])

        # Save GMM and resampled pc
        gma = gmc.mixture.weights(gmbatch)
        gmp = gmc.mixture.positions(gmbatch)
        gmcov = gmc.mixture.covariances(gmbatch)
        data_loading.write_gm_to_ply(gma, gmp, gmcov, 0, os.path.join(gengmm_path, str(runid).zfill(9) + ".gma.ply"))
        # data_loading.write_pc_to_off(os.path.join(recpc_path, str(runid).zfill(9) + ".off"), reconstructed)

vis.finish()