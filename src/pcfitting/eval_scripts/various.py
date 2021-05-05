from pcfitting import data_loading, PCDatasetIterator
from pcfitting.error_functions import ReconstructionStats, ReconstructionStatsProjected
import torch, math
from pcfitting.cpp.gmeval import pyeval
import os

# # -- CALCULATE RMSD/RMSD-P VALUES ON GIVEN PCs

# recstat = ReconstructionStats()
# recstatproj = ReconstructionStatsProjected(recstat)
#
# modelfile = r"D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_diff_scales/models-onlyplane/plane0-original.off"
# original = data_loading.load_pc_from_off(r"D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_diff_scales/fitpcs/n100000/plane0-original.off")
# resampled = data_loading.load_pc_from_off(r"D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_diff_scales/resampled/n100000/210404-Plane1G/manual/plane0-original.off.gmm.ply.off")
# #resampled = data_loading.load_pc_from_off(r"D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_diff_scales/resampled/n100000/210405-Plane1G-small/manual/plane0-original-SMALL.off.gmm.ply.off")
#
# names = recstat.get_names()
# loss = recstat.calculate_score_on_reconstructed(original, resampled, modelfile)
# for k in range(len(names)):
#     print("  ", names[k], ": ", loss[k].item())
# names = recstatproj.get_names()
# loss2 = recstatproj.calculate_score_on_reconstructed(original, resampled, modelfile)
# for k in range(len(names)):
#     print("  ", names[k], ": ", loss[k].item())

# # -- CALCULATE AVERAGE DISTANCE IN UNIFORM PC --
#
# n_points = 500000
# iterations = 60
# for i in range(iterations):
#     random = torch.rand(n_points, 2) * 128
#     pc = torch.zeros(n_points, 3)
#     pc[:, 0:2] = random
#     md = pyeval.calc_rmsd_to_itself(pc)[1]
#     print(md)

# # -- COMPARE NN-NORM AND AR-NORM --
#
# import trimesh
# import trimesh.sample
# import math
# from pcfitting.cpp.gmeval import pyeval
# #modelpath = r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\data\dataset_planes\models\airplane_0001.off"
# modelpath = r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\data\dataset_diff_scales\models\bed_0001.off"
# mesh = trimesh.load_mesh(modelpath)
#
# for pointcount in [100, 1000, 10000, 100000, 500000]:
#     samples, _ = trimesh.sample.sample_surface(mesh, pointcount)
#     pc = torch.from_numpy(samples)
#     md = pyeval.calc_rmsd_to_itself(pc)[1]
#     refdist = 128 / (2 * math.sqrt(pc.shape[0]) - 1)
#     print("NN-Faktor für ", pointcount, " Punkte: ", (refdist / md))
#
# print("AR-Faktor: ", (128 / math.sqrt(mesh.area)))

# # -- CALCULATE BB-NORM --
# model_path = r"F:\DA-Eval\dataset20\models"
# evalpc_path = r"F:\DA-Eval\dataset20\evalpcs"
# iterator = PCDatasetIterator(1, 100000, evalpc_path, model_path)
# linear_factors = []
# density_factors = []
# log_factors = []
# while iterator.has_next():
#     pcbatch, names = iterator.next_batch()
#     # pcbatch = pcbatch[0]
#     # pmax = pcbatch.max(dim=0)[0]
#     # pmin = pcbatch.min(dim=0)[0]
#     # blen = torch.norm(pmax - pmin).item()
#     # linear_factor = blen / math.sqrt(3)
#     md = pyeval.calc_rmsd_to_itself(pcbatch.view(-1, 3))[1]
#     refdist = 128 / (2 * math.sqrt(pcbatch.shape[1]) - 1)
#     linear_factor = refdist / md
#     density_factor = math.pow(linear_factor, -3)
#     log_factor = -3 * math.log(linear_factor)
#     linear_factors.append(linear_factor)
#     density_factors.append(density_factor)
#     log_factors.append(log_factor)
#
# for i in range(len(linear_factors)):
#     print(linear_factors[i])
# print("--")
# for i in range(len(density_factors)):
#     print(density_factors[i])
# print("--")
# for i in range(len(log_factors)):
#     print(log_factors[i])
# print("--")

# # -- CALCULATE KURTOSIS --
# pc = data_loading.load_pc_from_off(r"F:\DA-Eval\dataset20\evalpcs\n100000\curtain_0001.off")
# gm = data_loading.read_gm_from_ply(r"F:\DA-Eval\dataset20\gmms-significance\000000058.gma.ply", ismodel=False)
#
# evaldistance = ReconstructionStats()
# distvalues = evaldistance.calculate_score_packed(pc, gm)
#
# print (distvalues.rmsd_pure_I)
# print (distvalues.kurtosisI)

# -- BENCHMARK RMSD CALCULATION --

from pcfitting import data_loading, GMSampler
import gmc.mixture
import time

pcbatch = data_loading.load_pc_from_off(r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\data\dataset_diff_scales\evalpcs\n100000\bed_0001.off")
#mix = data_loading.read_gm_from_ply(r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\data\dataset_diff_scales\gmms\210306-01-EmEckPre\EMfps\bed_0001.off.gma.ply", ismodel=False)
mix = data_loading.read_gm_from_ply(r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\data\dataset_diff_scales\gmms\210306-01-EmEckPre\Preiner-0.9-5\bed_0001.off.gma.ply", ismodel=False)
gmm = gmc.mixture.convert_amplitudes_to_priors(mix)
t1 = time.time()
sampled = GMSampler.sampleGMM(gmm, 100000)
t2 = time.time()
print("Sampling (Py):  ", (t2 - t1))
t1 = time.time()
sampled2 = GMSampler.sampleGMM_ext(gmm, 100000)
t2 = time.time()
print("Sampling (C++): ", (t2 - t1))
#Damn, this also takes a lot of time
data_loading.write_pc_to_off(r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\data\dataset_diff_scales\resampled\resP-py.off", sampled)
t1 = time.time()
data_loading.write_pc_to_off(r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\data\dataset_diff_scales\resampled\resP-cp.off", sampled2)
t2 = time.time()
print("Writing      :  ", (t2 - t1))

t1 = time.time()
rmsd, md, stdev, maxd = pyeval.eval_rmsd_unscaled(pcbatch.view(-1, 3), sampled.view(-1, 3))
t2 = time.time()
print(rmsd)
print("Evaluation(1): ", (t2 - t1))
t1 = time.time()
rmsd, md, stdev, maxd = pyeval.eval_rmsd_unscaled(pcbatch.view(-1, 3), sampled2.view(-1, 3))
t2 = time.time()
print(rmsd)
print("Evaluation(2): ", (t2 - t1))
# print("Evaluation: ", (t3 - t2))
# print("Total:      ", (t3 - t1))

# Sampling:    29.20902156829834
# Evaluation:  6.024001121520996
# Total:       35.233022689819336

# Sampling (Py):   29.334959030151367
# Sampling (C++):  2.0239968299865723