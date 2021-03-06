# ------------------------------------------------
# Script for executing the experiments for the GMM-evaluation
# Can be deleted if not needed anymore
# ------------------------------------------------

import os

from pcfitting.eval_scripts.eval_db_access_v2 import EvalDbAccessV2
from pcfitting import programs, MaxIterationTerminationCriterion, RelChangeTerminationCriterion, PCDatasetIterator, \
    data_loading, GMSampler
from pcfitting.generators import GradientDescentGenerator, EMGenerator, EckartGeneratorSP, EckartGeneratorHP, PreinerGenerator
from pcfitting.error_functions import AvgDensities, ReconstructionStats, GMMStats, Smoothness, ReconstructionStatsFiltered, ReconstructionStatsProjected
import time
import gmc.mixture
from gmc.cpp.gm_vis.gm_vis import GMVisualizer
import matplotlib
import matplotlib.image as mimg
import gc
matplotlib.use('TkAgg')

model_path = r"K:\DA-Eval\dataset_eval_big\models"
fitpc_path = r"K:\DA-Eval\dataset_eval_big\fitpcs"
evalpc_path = r"K:\DA-Eval\dataset_eval_big\evalpcs"
recpc_path = r"K:\DA-Eval\dataset_eval_big\recpcs"
gengmm_path = r"K:\DA-Eval\dataset_eval_big\gmms"
rendering_path = r"K:\DA-Eval\dataset_eval_big\renderings"
db_path = r"K:\DA-Eval\EvalV3.db"

initterm = MaxIterationTerminationCriterion(0)
terminator2 = RelChangeTerminationCriterion(0.1, 20)

#em_params = [(1e-5, 64, 'fpsmax'), (1e-5, 256, 'fpsmax'), (1e-5, 1024, 'fpsmax')]
#em_params = [(1e-5, 512, 'randnormpos')]
em_params = []
eck_params = []
#eck_params = [(1e-5, 8, 3, 'fpsmax', 0.0)]
# eck_params = [(1e-5, 8, 3, 'randnormpos', 0.1)]
# 64, 64, 256, 1024
# eck_params = [(1e-5, 4, 3, 'fpsmax', 0.1), (1e-5, 8, 2, 'fpsmax', 0.1), (1e-5, 4, 4, 'fpsmax', 0.1), (1e-5, 4, 5, 'fpsmax', 0.1)]
eck_sp_params = [(1e-5, 8, 3, 'eigen', 0.1)]
eck_hp_params = []#[(1e-5, 8, 3, 'eigen')]
# 4096
# eck_params = [(1e-5, 4, 6, 'fpsmax', 0.1), (1e-5, 8, 4, 'fpsmax', 0.1)]
#init_params = [(1e-4, 512), (1e-5, 512), (1e-6, 512), (1e-7, 512)]
#init_params = [(1e-5, 512)]
init_params = []
pre_params = []
#pre_params = [(512, 0.8, 4, 1), (512, 0.9, 4, 1), (512, 1.0, 4, 1), (512, 1.1, 4, 1)]
#pre_params = [(512, 1.2, 4, 1), (512, 1.3, 4, 1), (512, 1.0, 5, 1), (512, 1.0, 3, 1), (512, 1.0, 2, 1), (512, 1.0, 1, 1)]
#pre_params = [(4000, 1.1, 4, 0)]
#pre_params = [(512, 1.1, 1, 1), (512, 1.1, 2, 1), (512, 1.1, 3, 1), (512, 1.1, 5, 1)]
#pre_params = [(64, 1.1, 4, 1), (256, 1.1, 4, 1), (1024, 1.1, 4, 1)]
#pre_params = [(64, 0.7, 4, 1), (8000, 1.1, 4, 1), (16000, 1.1, 4, 1), (32000, 1.1, 4, 1), (8000, 1.1, 4, 0), (16000, 1.1, 4, 0), (32000, 1.1, 4, 0)]


generators = []

for (eps, ng, init) in em_params:
    generators.append(EMGenerator(n_gaussians=ng, initialization_method=init, termination_criterion=terminator2, em_step_points_subbatchsize=10000, verbosity=0, eps=eps))

for (eps, j, l, init, thresh) in eck_sp_params:
    generators.append(EckartGeneratorSP(n_gaussians_per_node=j, n_levels=l, partition_threshold=thresh, termination_criterion=terminator2, initialization_method=init, m_step_points_subbatchsize=10000, m_step_gaussians_subbatchsize=512,
                            e_step_pair_subbatchsize=5120000, eps=eps))
for (eps, j, l, init) in eck_hp_params:
    generators.append(EckartGeneratorHP(n_gaussians_per_node=j, n_levels=l, m_step_points_subbatchsize=10000,
                                        initialization_method=init, termination_criterion=terminator2, eps=eps))

for (eps, ng) in init_params:
    generators.append(EMGenerator(n_gaussians=ng, initialization_method='fpsmax', termination_criterion=initterm,
                                  em_step_points_subbatchsize=10000, verbosity=0, eps=eps))

for (ng, fd, alpha, ao) in pre_params:
    generators.append(PreinerGenerator(ngaussians=ng, alpha=alpha, fixeddist=fd, avoidorphansmode=ao, verbosity=0))

ns_fit_points = [50000]
n_eval_points_density = 1000000
n_eval_points_distance = 100000

vis = GMVisualizer(False, 800, 800)
vis.set_camera_auto(True)
vis.set_density_rendering(True)
vis.set_ellipsoids_pc_rendering(False, False, False)
vis.set_whitemode(True)

for n_fit_points in ns_fit_points:
    dataset_fit = PCDatasetIterator(1, n_fit_points, fitpc_path, model_path)
    dataset_eval_dens = PCDatasetIterator(1, n_eval_points_density, evalpc_path, model_path)
    dataset_eval_dist = PCDatasetIterator(1, n_eval_points_distance, evalpc_path, model_path)
    dbaccess = EvalDbAccessV2(db_path)
    evaldensity = AvgDensities()
    evalsmooth = Smoothness(subsamples=10000)
    evaldistane = ReconstructionStats()
    evalfilproj = ReconstructionStatsFiltered(ReconstructionStatsProjected(ReconstructionStats(rmsd_scaled_by_nn=False, md_scaled_by_nn=False, cv=True, inverse=False, chamfer_norm_nn=False, sample_points=n_eval_points_distance)))
    evalstats = GMMStats()

    batchcount = dataset_fit.remaining_batches_count()

    i = -1
    while dataset_fit.has_next():
        i += 1
        batch, names = dataset_fit.next_batch()
        batch_eval_dens, _ = dataset_eval_dens.next_batch()
        batch_eval_dist, _ = dataset_eval_dist.next_batch()
        nns = dbaccess.get_nn_scale_factor(names[0])
        batch.nnscalefactor = nns
        batch_eval_dens.nnscalefactor = nns
        batch_eval_dist.nnscalefactor = nns
        for j in range(len(generators)):
            print(100 * ((i / batchcount) + (j / len(generators) / batchcount)), "%")
            print("Generator ", (j+1), "/", len(generators), " on ", names)

            exists = False
            if isinstance(generators[j], EMGenerator):
                exists = dbaccess.has_em_run(names[0], n_fit_points, generators[j]._n_gaussians, n_fit_points,
                                   str(generators[j]._termination_criterion),
                                   generators[j]._initialization_method, "float32", generators[j]._epsvar, True)
            elif isinstance(generators[j], EckartGeneratorSP):
                exists = dbaccess.has_eck_sp_run(names[0], n_fit_points, generators[j]._n_gaussians_per_node ** generators[j]._n_levels,
                                        generators[j]._n_gaussians_per_node, generators[j]._n_levels, generators[j]._partition_threshold,
                                        str(generators[j]._termination_criterion), generators[j]._initialization_method, "float32", generators[j]._epsvar, True)
            elif isinstance(generators[j], EckartGeneratorHP):
                exists = dbaccess.has_eck_hp_run(names[0], n_fit_points,
                                       generators[j]._n_gaussians_per_node ** generators[j]._n_levels,
                                       generators[j]._n_gaussians_per_node, generators[j]._n_levels,
                                       str(generators[j]._termination_criterion), generators[j]._initialization_method,
                                       "float32", generators[j]._epsvar, True)
            elif isinstance(generators[j], PreinerGenerator):
                p = generators[j]._params
                exists = dbaccess.has_preiner_run(names[0], n_fit_points, p.ngaussians, p.alpha, p.pointpos, p.stdev,
                                                  p.iso, p.inittype, p.knn, p.fixeddist, p.weighted, p.levels,
                                                  p.reductionfactor, p.ngaussians, p.avoidorphans)
            else:
                print("Unknown generator")
                exit(-1)
            if exists:
                print("Skip")
                continue

            modelpath = os.path.join(model_path, names[0])

            print ("Fitting")

            start = time.time()

            gmbatch, gmmbatch = generators[j].generate(batch)

            end = time.time()

            # Evaluate
            print ("Evaluating")
            densvalues_eval = evaldensity.calculate_score_packed(batch_eval_dens, gmbatch, modelpath=modelpath)
            smoothvalues = evalsmooth.calculate_score_packed(batch_eval_dens, gmbatch, modelpath=modelpath)
            statvalues = evalstats.calculate_score_packed(batch, gmbatch, modelpath=modelpath)
            reconstructed = GMSampler.sampleGMM_ext(gmmbatch, n_eval_points_distance)
            distvalues = evaldistane.calculate_score_on_reconstructed(batch_eval_dist, reconstructed, modelpath=modelpath)
            projvalues = evalfilproj.calculate_score_on_reconstructed(batch_eval_dist, reconstructed, modelpath=modelpath)

            # Render
            print ("Rendering")
            vis.set_pointclouds(reconstructed.cpu())
            vis.set_gaussian_mixtures(gmbatch.cpu())
            res = vis.render()

            # Save in DB
            print ("Saving")
            #EM
            if isinstance(generators[j], EMGenerator):
                emid = dbaccess.insert_options_em(n_fit_points, str(generators[j]._termination_criterion), generators[j]._initialization_method,
                                              "float32", generators[j]._epsvar, True)
                runid = dbaccess.insert_run(names[0], n_fit_points, generators[j]._n_gaussians, gmbatch.shape[2],
                                        "EM", emid, (end - start))
            #Eckart
            elif isinstance(generators[j], EckartGeneratorSP):
                eckid = dbaccess.insert_options_eckart_sp(generators[j]._n_gaussians_per_node, generators[j]._n_levels,
                                                       generators[j]._partition_threshold,
                                                       str(generators[j]._termination_criterion),
                                                       generators[j]._initialization_method, "float32",
                                                       generators[j]._epsvar, True)
                runid = dbaccess.insert_run(names[0], n_fit_points, generators[j]._n_gaussians_per_node ** generators[j]._n_levels,
                                        gmbatch.shape[2], "EckSP", eckid, (end - start))
            elif isinstance(generators[j], EckartGeneratorHP):
                eckid = dbaccess.insert_options_eckart_hp(generators[j]._n_gaussians_per_node, generators[j]._n_levels,
                                                      str(generators[j]._termination_criterion),
                                                      generators[j]._initialization_method, "float32",
                                                      generators[j]._epsvar, True)
                runid = dbaccess.insert_run(names[0], n_fit_points, generators[j]._n_gaussians_per_node ** generators[j]._n_levels,
                                        gmbatch.shape[2], "EckHP", eckid, (end - start))
            elif isinstance(generators[j], PreinerGenerator):
                p = generators[j]._params
                preid = dbaccess.insert_options_preiner(p.alpha, p.pointpos, p.stdev, p.iso, p.inittype, p.knn, p.fixeddist,
                                                        p.weighted, p.levels, p.reductionfactor, p.ngaussians,
                                                        p.avoidorphans)
                runid = dbaccess.insert_run(names[0], n_fit_points, p.ngaussians, gmbatch.shape[2], "Preiner", preid,
                                            (end - start))
            else:
                print("Unknown generator")
                exit(-1)

            dbaccess.insert_eval_density(densvalues_eval.logavg_scaled_nn, densvalues_eval.logstdv,
                                         densvalues_eval.avg_scaled_nn, densvalues_eval.stdev_scaled_nn, densvalues_eval.cv,
                                         smoothvalues[0].item(), smoothvalues[1].item(), runid)
            dbaccess.insert_eval_distance(distvalues.rmsd_scaled_by_nn, distvalues.md_scaled_by_nn, distvalues.stdev_scaled_by_nn, distvalues.cv,
                                          distvalues.rmsd_scaled_by_nn_I, distvalues.md_scaled_by_nn_I, distvalues.stdev_scaled_by_nn_I, distvalues.cv_I,
                                          distvalues.rcd_norm_nn, projvalues.stdev_scaled_by_nn, projvalues.cv, runid)
            dbaccess.insert_eval_stat(statvalues[0].item(), statvalues[1].item(), statvalues[2].item(),
                                      statvalues[3].item(), statvalues[4].item(), statvalues[5].item(),
                                      statvalues[6].item(), statvalues[7].item(), statvalues[8].item(),
                                      statvalues[9].item(), statvalues[10].item(), statvalues[11].item(),
                                      statvalues[12].item(), statvalues[13].item(), statvalues[14].item(),
                                      statvalues[15].item(), statvalues[16].item(), statvalues[17].item(),
                                      statvalues[18].item(), runid, statvalues[19].item(), statvalues[20].item(),
                                      statvalues[21].item())

            #mimg.imsave(os.path.join(rendering_path, "recpc-" + str(runid).zfill(9) + ".png"), res[0, 0])
            mimg.imsave(os.path.join(rendering_path, "density-" + str(runid).zfill(9) + ".png"), res[0, 0])

            # Save GMM and resampled pc
            gma = gmc.mixture.weights(gmbatch)
            gmp = gmc.mixture.positions(gmbatch)
            gmcov = gmc.mixture.covariances(gmbatch)
            data_loading.write_gm_to_ply(gma, gmp, gmcov, 0, os.path.join(gengmm_path, str(runid).zfill(9) + ".gma.ply"))
            # data_loading.write_pc_to_off(os.path.join(recpc_path, str(runid).zfill(9) + ".off"), reconstructed)

        gc.collect()

vis.finish()