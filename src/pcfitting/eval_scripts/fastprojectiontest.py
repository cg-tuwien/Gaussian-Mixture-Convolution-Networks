# ------------------------------------------------
# Script for comparing execution times and results of certain operations with different libraries or methods
# Can be deleted if not needed anymore
# ------------------------------------------------

import pytorch3d.loss.point_mesh_distance as pmd
from pytorch3d import loss
from pytorch3d.structures import Meshes, Pointclouds
import trimesh
from pcfitting import data_loading, GMSampler
from pcfitting.error_functions import ReconstructionStats, ReconstructionStatsProjected, RcdLoss
from pytorch3d import _C
import torch
import time
from pysdf import SDF
import numpy as np
import open3d as o3d
import trimesh.proximity
import gmc.mixture

model_path = r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\data\dataset_diff_scales\models\bed_0001.off"
trim = trimesh.load_mesh(model_path)
faces = torch.tensor(trim.faces).view(1, -1, 3).cuda()
verts = torch.tensor(trim.vertices).view(1, -1, 3).cuda()
rfaces = verts[0, faces]
#mesh = Meshes(torch.tensor(trim.vertices).view(1, -1, 3), torch.tensor(trim.faces).view(1, -1, 3))

evalpc = data_loading.load_pc_from_off(r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\data\dataset_diff_scales\evalpcs\n100000\bed_0001.off")

gm = data_loading.read_gm_from_ply(r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\data\dataset_diff_scales\gmms\210918-04-EMinit\fps\bed_0001.off.gma.ply", False)
#gm = data_loading.read_gm_from_ply(r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\da-gm-1\da-gm-1\data\testgmellip.ply", True)
gcount = gm.shape[2]
gmm_temp = gmc.mixture.pack_mixture(torch.ones(1, 1, gcount).cuda() / gcount, gmc.mixture.positions(gm), gmc.mixture.covariances(gm))

gmx = gmc.mixture.convert_priors_to_amplitudes(gmm_temp)
recpcx = GMSampler.sampleGM_ext(gmx, 100000)
recpc = GMSampler.sampleGM_ext(gm, 100000)

# Calculating chamfer distance using PyTorch 3D
time1 = time.time()
chamfer = loss.chamfer_distance(recpcx, evalpc)[0]
time2 = time.time()
print(chamfer.item(), (time2 - time1))

# Calculating chamfer distance using our RcdLoss
time1 = time.time()
rcdloss = RcdLoss(100000, gcount)
cl = rcdloss.calculate_score_packed(evalpc, gmx)
time2 = time.time()
print (cl.item(), (time2 - time1))

exit()

# Calculating chamfer distance using PySDF
time1 = time.time()
evn = evalpc.view(-1, 3).cpu().numpy()
rcn = recpc.view(-1, 3).cpu().numpy()
sdfE = SDF(evn, np.zeros((0, 3)))
#d1 = sdfE(rcn)
d1 = torch.linalg.norm(torch.tensor(rcn - evn[sdfE.nn(rcn)]),dim=1)
sdfR = SDF(rcn, np.zeros((0, 3)))
d2 = torch.linalg.norm(torch.tensor(evn - rcn[sdfR.nn(evn)]),dim=1)
rmsd1 = (torch.tensor(d1) ** 2).mean()
rmsd2 = (torch.tensor(d2) ** 2).mean()
ch3 = rmsd1 + rmsd2
time2 = time.time()
print(ch3.item(), (time2-time1))

# Calculating chamfer distance using our own implementation
stats = ReconstructionStats(rmsd_scaled_by_nn=False, md_scaled_by_nn=False, stdev_scaled_by_nn=False, cv=False, inverse=False, chamfer=True, chamfer_norm_nn=False, usepysdf=False)
time1 = time.time()
ch2 = stats.calculate_score_on_reconstructed(evalpc, recpc)
time2 = time.time()
print(ch2[0].item(), (time2-time1))
# PyTorch3d is much faster

recpcN = recpc.view(-1,3).cpu().numpy()

# Project points using TriMesh
time1 = time.time()
mesh = trimesh.load_mesh(model_path)
query = trimesh.proximity.ProximityQuery(mesh)
closest, distance, triangle_id = query.on_surface(recpcN)
pointsTrimesh = torch.from_numpy(closest).cuda()
time2 = time.time()
print("TriMesh", time2-time1)

# Project points using Open3D
time1 = time.time()
meshO = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(model_path))
query_point = o3d.core.Tensor(recpcN, dtype=o3d.core.Dtype.Float32)
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(meshO)
ans = scene.compute_closest_points(query_point)
pointsOpen3D = torch.from_numpy(ans['points'].numpy()).cuda()
time2 = time.time()
print("Open3D", time2-time1)

# Project points with our method (now also Open3D)
print ("--")
time1 = time.time()
stats = ReconstructionStatsProjected(ReconstructionStats(rmsd_scaled_by_nn=False, md_scaled_by_nn=False, stdev_scaled_by_nn=True, cv=False, inverse=False, chamfer_norm_nn=False, usepysdf=False))
std1 = stats.calculate_score_on_reconstructed(evalpc, recpc, modelpath=model_path)
time2 = time.time()
print(std1[0].item(), (time2-time1))

# With fewer points
time1 = time.time()
# evn = evalpc.view(-1, 3).cpu().numpy()
# rcn = recpc.view(-1, 3).cpu().numpy()
# sdfR = SDF(rcn, np.zeros((0, 3)))
# d2 = torch.linalg.norm(torch.tensor(evn - rcn[sdfR.nn(evn)]),dim=1)
# std2 = d2.std()
stats = ReconstructionStatsProjected(ReconstructionStats(rmsd_scaled_by_nn=False, md_scaled_by_nn=False, stdev_scaled_by_nn=True, cv=False, inverse=False, chamfer_norm_nn=False, usepysdf=True, sample_points=50000))
std2 = stats.calculate_score_on_reconstructed(evalpc, recpc, modelpath=model_path)
time2 = time.time()
print(std2.item(), (time2-time1))


print("test")