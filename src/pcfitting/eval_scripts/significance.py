from pcfitting import MaxIterationTerminationCriterion, RelChangeTerminationCriterion, PCDatasetIterator, GMSampler, data_loading
from pcfitting.generators import EMGenerator, EckartGeneratorSP, PreinerGenerator
from pcfitting.error_functions import AvgDensities, ReconstructionStats, ReconstructionStatsProjected, GMMStats
from gmc.cpp.gm_vis.gm_vis import GMVisualizer
import matplotlib
import matplotlib.image as mimg
matplotlib.use('TkAgg')
import gmc.mixture
import sqlite3
import os

model_path = r"F:\DA-Eval\dataset20\models"
fitpc_path = r"F:\DA-Eval\dataset20\fitpcs"
evalpc_path = r"F:\DA-Eval\dataset20\evalpcs"
recpc_path = r"F:\DA-Eval\dataset20\recpcs-significance"
gengmm_path = r"F:\DA-Eval\dataset20\gmms-significance"
rendering_path = r"F:\DA-Eval\dataset20\renderings-significance"
dbpath = r"F:\DA-Eval\Significance.db"

generators = [
    EMGenerator(n_gaussians=512, initialization_method="fpsmax", termination_criterion=MaxIterationTerminationCriterion(0), em_step_points_subbatchsize=10000),
    EMGenerator(n_gaussians=512, initialization_method="randnormpos", termination_criterion=RelChangeTerminationCriterion(0.1, 20), em_step_points_subbatchsize=10000),
    EMGenerator(n_gaussians=512, initialization_method="fpsmax", termination_criterion=RelChangeTerminationCriterion(0.1, 20), em_step_points_subbatchsize=10000, eps=1e-5),
    EMGenerator(n_gaussians=512, initialization_method="fpsmax", termination_criterion=RelChangeTerminationCriterion(0.1, 20), em_step_points_subbatchsize=10000, eps=1e-7),
    EMGenerator(n_gaussians=512, initialization_method="fpsmax", termination_criterion=RelChangeTerminationCriterion(0.1, 20), em_step_points_subbatchsize=10000, eps=1e-9),
    EckartGeneratorSP(n_gaussians_per_node=8, n_levels=3, termination_criterion=RelChangeTerminationCriterion(0.1, 20), initialization_method="fpsmax", partition_threshold=0.1, m_step_points_subbatchsize=10000, e_step_pair_subbatchsize=5120000),
    PreinerGenerator(fixeddist=0.9, ngaussians=512, alpha=5, avoidorphans=False)
]

generator_identifiers = ["Init", "EMrnp", "EMfps-5", "EMfps-7", "EMfps-9", "Eckart", "Preiner"]

n_fit_points = 100000
n_eval_points_density = 1000000
n_eval_points_distance = 100000

dataset_fit = PCDatasetIterator(1, n_fit_points, fitpc_path, model_path)
dataset_eval_dens = PCDatasetIterator(1, n_eval_points_density, evalpc_path, model_path)
dataset_eval_dist = PCDatasetIterator(1, n_eval_points_distance, evalpc_path, model_path)
db = sqlite3.connect(dbpath)
evaldensity = AvgDensities()
evaldistance = ReconstructionStats()
evaldistanceP = ReconstructionStatsProjected()
evalstats = GMMStats()
vis = GMVisualizer(False, 800, 800)
vis.set_camera_auto(True)
vis.set_density_rendering(True)
vis.set_ellipsoids_pc_rendering(False, True, False)

batchcount = dataset_fit.remaining_batches_count()

starti = 13
startj = 0

i = -1
while dataset_fit.has_next():
    i += 1
    batch, names = dataset_fit.next_batch()
    batch_eval_dens, _ = dataset_eval_dens.next_batch()
    batch_eval_dist, _ = dataset_eval_dist.next_batch()
    if i < starti:
        continue
    for j in range(len(generators)):
        if i == starti and j < startj:
            continue
        print(100 * ((i / batchcount) + (j / len(generators) / batchcount)), "%")
        print("Generator ", generator_identifiers[j], " on ", names, " / #", i, "-", j)

        modelpath = os.path.join(model_path, names[0])

        print ("Fitting")

        gmbatch, gmmbatch = generators[j].generate(batch)

        print ("Evaluating")
        densvalues_eval = evaldensity.calculate_score_packed(batch_eval_dens, gmbatch, modelpath=modelpath)
        statvalues = evalstats.calculate_score_packed(batch, gmbatch, modelpath=modelpath)
        reconstructed = GMSampler.sampleGMM(gmmbatch, n_eval_points_distance)
        distvalues = evaldistance.calculate_score_on_reconstructed(batch_eval_dist, reconstructed, modelpath=modelpath)
        distvaluesP = evaldistanceP.calculate_score_on_reconstructed(batch_eval_dist, reconstructed, modelpath=modelpath)

        print ("Rendering")
        vis.set_pointclouds(reconstructed.cpu())
        vis.set_gaussian_mixtures(gmbatch.cpu())
        res = vis.render()

        print ("Saving")

        sql1 = "INSERT INTO Run (algorithm, model, mu_L, mu_D, sigma_D, v_D, RMSD, MD, STD, CV, RMSDI, MDI, STDI, CVI, COVM, COVMSTD, RMSDP, MDP, STDP, CVP, RMSDIP, MDIP, STDIP, CVIP, COVMP, COVMSTDP, min_ev)" \
               "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        cur = db.cursor()
        cur.execute(sql1, (
            generator_identifiers[j], names[0], densvalues_eval.logavg, densvalues_eval.avg, densvalues_eval.stdev,
            densvalues_eval.cv,
            distvalues.rmsd_pure, distvalues.md_pure, distvalues.stdev, distvalues.cv,
            distvalues.rmsd_pure_I, distvalues.md_pure_I, distvalues.stdev_I, distvalues.cv_I,
            distvalues.cov_measure, distvalues.cov_measure_std,
            distvaluesP.rmsd_pure, distvaluesP.md_pure, distvaluesP.stdev, distvaluesP.cv,
            distvaluesP.rmsd_pure_I, distvaluesP.md_pure_I, distvaluesP.stdev_I, distvaluesP.cv_I,
            distvaluesP.cov_measure, distvaluesP.cov_measure_std,
            statvalues[8, 0].item()
        ))
        db.commit()
        runid = cur.lastrowid
        sql2 = "INSERT INTO Run_AR (ID, algorithm, model, mu_L, mu_D, sigma_D, v_D, RMSD, MD, STD, CV, RMSDI, MDI, STDI, CVI, COVM, COVMSTD, RMSDP, MDP, STDP, CVP, RMSDIP, MDIP, STDIP, CVIP, COVMP, COVMSTDP, min_ev)" \
               "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        cur.execute(sql2, (runid,
            generator_identifiers[j], names[0], densvalues_eval.logavg_scaled_area, densvalues_eval.avg_scaled_area, densvalues_eval.stdev_scaled_area,
            densvalues_eval.cv,
            distvalues.rmsd_scaled_by_area, distvalues.md_scaled_by_area, distvalues.stdev_scaled_by_area, distvalues.cv,
            distvalues.rmsd_scaled_by_area_I, distvalues.md_scaled_by_area_I, distvalues.stdev_scaled_by_area_I, distvalues.cv_I,
            distvalues.cov_measure, distvalues.cov_measure_std_scaled_by_area,
            distvaluesP.rmsd_scaled_by_area, distvaluesP.md_scaled_by_area, distvaluesP.stdev_scaled_by_area, distvaluesP.cv,
            distvaluesP.rmsd_scaled_by_area_I, distvaluesP.md_scaled_by_area_I, distvaluesP.stdev_scaled_by_area_I, distvaluesP.cv_I,
            distvaluesP.cov_measure, distvaluesP.cov_measure_std_scaled_by_area,
            statvalues[8, 0].item()
        ))
        db.commit()

        mimg.imsave(os.path.join(rendering_path, "recpc-" + str(runid).zfill(9) + ".png"), res[0, 0])
        mimg.imsave(os.path.join(rendering_path, "density-" + str(runid).zfill(9) + ".png"), res[0, 1])

        # Save GMM and resampled pc
        gma = gmc.mixture.weights(gmbatch)
        gmp = gmc.mixture.positions(gmbatch)
        gmcov = gmc.mixture.covariances(gmbatch)
        data_loading.write_gm_to_ply(gma, gmp, gmcov, 0, os.path.join(gengmm_path, str(runid).zfill(9) + ".gma.ply"))
        data_loading.write_pc_to_off(os.path.join(recpc_path, str(runid).zfill(9) + ".off"), reconstructed)

vis.finish()