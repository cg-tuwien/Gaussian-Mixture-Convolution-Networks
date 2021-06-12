import time
from pcfitting import MaxIterationTerminationCriterion, RelChangeTerminationCriterion, PCDatasetIterator, GMSampler, data_loading
from pcfitting.generators import EMGenerator, EckartGeneratorSP, PreinerGenerator
from pcfitting.error_functions import AvgDensities, ReconstructionStats, ReconstructionStatsProjected, GMMStats, IntUniformity, Smoothness
from gmc.cpp.gm_vis.gm_vis import GMVisualizer
import matplotlib
import matplotlib.image as mimg
matplotlib.use('TkAgg')
import gmc.mixture
import sqlite3
import os
from pcfitting.cpp.gmeval import pyeval

evalpc_path = r"F:\DA-Eval\dataset_eval\evalpcs\n100000\bed_0001.off"
# evalpc_path = r"F:\DA-Eval\dataset_eval\evalpcs\n1000000\bed_0001.off"
gmm_path = r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\thesis\EvalDraft\images\gmms"
model_path = r"F:\DA-Eval\dataset_eval\models\bed_0001.off"

evalpc = data_loading.load_pc_from_off(evalpc_path).view(-1, 3)

uni = IntUniformity(True, False, True)
avgd = AvgDensities(calculate_logavg=True)
rec = ReconstructionStats(chamfer=True, stdev=True, cov_measure=True)
recp = ReconstructionStatsProjected(rec)
smoothness = Smoothness()
stats = GMMStats()
# names = uni.get_names()
# names = avgd.get_names()
# names = rec.get_names()
# names = smoothness.get_names()
# names = recp.get_names()
names = stats.get_names()

for root, dirs, files in os.walk(gmm_path):
    for name in files:
        if name.lower().endswith(".gma.ply"):
            path = os.path.join(root, name)
            relpath = path[len(gmm_path) + 1:]
            print(relpath)
            gm = data_loading.read_gm_from_ply(path, False)
            #genpc = GMSampler.sampleGM_ext(gm, 100000).view(-1, 3)
            # rmsdI, mdI, stdevI, maxdI = pyeval.eval_rmsd_unscaled(evalpc, genpc)
            # print("Std: ", stdevI)
            # print("Cv : ", stdevI / mdI)
            # starttime = time.time()
            # std1, cv1, std5, cv5 = pyeval.calc_std_1_5(evalpc, genpc)
            # endtime = time.time()
            # print("Std1: ", std1)       # ~5.9seconds
            # print("Std5: ", std5)       # 334 seconds ~ 5:30 minutes
            # print("Cv1 : ", cv1)
            # print("Cv5 : ", cv5)
            # starttime = time.time()
            # avgkl = pyeval.avg_kl_div(gmc.mixture.convert_amplitudes_to_priors(gm)[0,0].cpu())
            # endtime = time.time()
            # print(relpath, ": ", avgkl)
            # covmcv, covmst = pyeval.cov_measure(genpc)
            # print("CMC1: ", covmcv)
            # print("CMS1: ", covmst)
            # covmcv5, covmst5 = pyeval.cov_measure_5(genpc)
            # print("CMC5: ", covmcv5)
            # print("CMS5: ", covmst5)
            # res = uni.calculate_score_packed(evalpc, gm, None, model_path)
            epc = evalpc.unsqueeze(0)
            epc.nnscalefactor = 1
            starttime = time.time()
            # res = avgd.calculate_score_packed(epc, gm, None, model_path)
            # res = rec.calculate_score_packed(evalpc.unsqueeze(0), gm, None, model_path)
            # res = smoothness.calculate_score_packed(evalpc.unsqueeze(0), gm, None, model_path)
            # res = recp.calculate_score_packed(epc, gm, None, model_path)
            res = stats.calculate_score_packed(epc, gm, None, model_path)
            endtime = time.time()
            print(endtime - starttime)
            for k in range(len(names)):
                print("  ", names[k], ": ", res[k].item())

