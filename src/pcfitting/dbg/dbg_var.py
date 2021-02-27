from gmc import mat_tools
from pcfitting import data_loading, GMLogger, Scaler
from pcfitting.generators import EMGenerator
from pcfitting.generators.em_tools import EMTools
from pcfitting.error_functions import LikelihoodLoss
from pcfitting.termination_criterion import MaxIterationTerminationCriterion
import torch
import gmc.mixture as gm

# terminator = MaxIterationTerminationCriterion(50)
# pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/pointclouds/n100000/bathtub_0001.off").to(torch.float64)
# scaler = Scaler()
# scaler.set_pointcloud_batch(pcbatch)
# pcscaled = scaler.scale_down_pc(pcbatch)
# n_gaussians = 512
# # pcbatch = torch.tensor([[[0.0, 0.0, 0.0], [0.5, 0.3, 0.0], [0.6, 0.4, 0.2], [0.1, 0.3, 0.3], [0.9, 0.9, 0.9]]], device='cuda')
# # n_gaussians = 2
# logger = GMLogger(["em"], "em", "", 0, n_gaussians, 1, 0, 0, 0, scaler=scaler)
# emgen = EMGenerator(n_gaussians=n_gaussians, initialization_method="randnormpos", em_step_points_subbatchsize=10000, n_sample_points=10000,
#                  termination_criterion=terminator, dtype=torch.float64)
# emgen.set_logging(logger)
# gmbatch, gmmbatch = emgen.generate(pcscaled)
# loss = LikelihoodLoss().calculate_score_packed(pcscaled, gmbatch)
# print("Eval Loss: ", torch.exp(-loss).item())
#
# pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/pointclouds/n10000/bathtub_0001.off").to(torch.float64)
# scaler = Scaler()
# scaler.set_pointcloud_batch(pcbatch)
# pcscaled = scaler.scale_down_pc(pcbatch)
# n_gaussians = 512
# # pcbatch = torch.tensor([[[0.0, 0.0, 0.0], [0.5, 0.3, 0.0], [0.6, 0.4, 0.2], [0.1, 0.3, 0.3], [0.9, 0.9, 0.9]]], device='cuda')
# # n_gaussians = 2
# logger = GMLogger(["em"], "em", "", 0, n_gaussians, 1, 0, 0, 0, scaler=scaler)
# emgen = EMGenerator(n_gaussians=n_gaussians, initialization_method="randnormpos", em_step_points_subbatchsize=10000, n_sample_points=10000,
#                  termination_criterion=terminator, dtype=torch.float64)
# emgen.set_logging(logger)
# gmbatch, gmmbatch = emgen.generate(pcscaled)
# loss = LikelihoodLoss().calculate_score_packed(pcscaled, gmbatch)
# print("Eval Loss: ", torch.exp(-loss).item())

# # Counting the valid GMs
# mixs = ["1609609626-EM-fpsmax-unscaled-full100k",
#                          "1609609626-EM-fpsmax-unscaled-sampled10kv100k",
#                          "1609609947-EM-fpsmax-scaled-full100k",
#                          "1609609947-EM-fpsmax-scaled-sampled10kv100k",
#                          "1609668149-EM-fpsmax-scaled-sampled50kv100k",
#                          "1609668252-EM-fpsmax-unscaled-sampled50kv100k",
#                          "1609668470-EM-fpsmax-unscaled-full-10k",
#                          "1609668929-EM-fpsmax-scaled-full-10k"]
# for i in mixs:
#     mix = data_loading.read_gm_from_ply("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/gmms/DebugEck/" + i + "/1.gma.ply", ismodel=False)
#     print(i, "valid gaussians: ", (~gm.weights(mix).eq(0)).sum().item())
# exit(0)

# Testcase for scale up loss
# pcbatch = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/pointclouds/n10000/chair_0001.off")
# mix = data_loading.read_gm_from_ply("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/logs/210101-EckartEval1-10.000(fullem)/00-EMiRNP/chair_0001.off/gmm-00000.gma.ply", ismodel=False)
# scaler = Scaler()
# scaler.set_pointcloud_batch(pcbatch)
# pcdown = scaler.scale_pc(pcbatch)
# mixdown = scaler.scale_gm(mix)
# llh = LikelihoodLoss(False)
# loss_upper = llh.calculate_score_packed(pcbatch, mix)
# loss_lower = llh.calculate_score_packed(pcdown, mixdown)
# loss_l_up = scaler.unscale_losses(loss_lower)
# print("Upper: ", loss_upper.item())
# print("Lower: ", loss_lower.item())
# print("LUpsc: ", loss_l_up.item())

# # Comparing EM on Scale
pcoriginal = data_loading.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/pointclouds/n10000/chair_0001.off")
mix = data_loading.read_gm_from_ply("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/gmms/210101-EckartEval1-10.000(fullem)/00-EMiRNP/chair_0001.off.gma.ply", ismodel=False)
# mix = data_loading.read_gm_from_ply("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/logs/210101-EckartEval1-10.000(fullem)/00-EMiRNP/chair_0001.off/gmm-00000.gma.ply", ismodel=False)
scaler = Scaler()
scaler.set_pointcloud_batch(pcoriginal)
pcdown = scaler.scale_pc(pcoriginal)
pcbatch = torch.cat((pcdown, pcoriginal), dim=0)
mixdown = scaler.scale_gm(mix)
initgmbatch = torch.cat((mixdown, mix), dim=0)
emgen = EMGenerator(512, termination_criterion=MaxIterationTerminationCriterion(0), n_sample_points=-1, eps=1e-7, eps_is_relative=True, dtype=torch.float32)
gms, _ = emgen.generate(pcbatch)#, initgmbatch.float())
gm2a = gms[0:1]
gm2b = gms[1:2]
gm2aup = scaler.unscale_gm(gm2a)
assert EMTools.find_valid_matrices(gm.covariances(gm2aup), mat_tools.inverse(gm.covariances(gm2aup))).all()
loss = LikelihoodLoss(False)
print("gm2a: ", scaler.unscale_losses(loss.calculate_score_packed(pcdown.float(), gm2a)).item())
print("gm2aup: ", loss.calculate_score_packed(pcoriginal.float(), gm2aup).item())
print("gm2b:   ", loss.calculate_score_packed(pcoriginal.float(), gm2b).item())
diff = (gm2aup - gm2b).mean()
print(diff)
data_loading.write_gm_to_ply(gm.weights(gm2aup), gm.positions(gm2aup), gm.covariances(gm2aup), 0, "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/gm2aup.gma.ply")
data_loading.write_gm_to_ply(gm.weights(gm2b), gm.positions(gm2b), gm.covariances(gm2b), 0, "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/gm2b.gma.ply")

# Unbatched way:
# gm2a, _ = emgen.generate(pcdown, mixdown.double())
# gm2b, _ = emgen.generate(pcoriginal, mix.double())
# gm2c, _ = emgen.generate(pcdown)
# gm2d, _ = emgen.generate(pcoriginal)
# gm2aup = scaler.unscale_gm(gm2a)
# gm2cup = scaler.unscale_gm(gm2c)
# loss = LikelihoodLoss(False)
# print("gm2aup: ", loss.calculate_score_packed(pcoriginal.double(), gm2aup).item())
# print("gm2b:   ", loss.calculate_score_packed(pcoriginal.double(), gm2b).item())
# print("gm2cup: ", loss.calculate_score_packed(pcoriginal.double(), gm2cup).item())
# print("gm2d:   ", loss.calculate_score_packed(pcoriginal.double(), gm2d).item())
# diff = (gm2aup - gm2b).mean()
# print(diff)
# data_loading.write_gm_to_ply(gm.weights(gm2aup), gm.positions(gm2aup), gm.covariances(gm2aup), 0, "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/gm2aup.gma.ply")
# data_loading.write_gm_to_ply(gm.weights(gm2b), gm.positions(gm2b), gm.covariances(gm2b), 0, "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/gm2b.gma.ply")
# data_loading.write_gm_to_ply(gm.weights(gm2cup), gm.positions(gm2cup), gm.covariances(gm2cup), 0, "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/gm2cup.gma.ply")
# data_loading.write_gm_to_ply(gm.weights(gm2d), gm.positions(gm2d), gm.covariances(gm2d), 0, "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/gm2d.gma.ply")
