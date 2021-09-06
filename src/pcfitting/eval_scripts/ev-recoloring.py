import os
from gmc.cpp.gm_vis.gm_vis import GMVisualizer, GmVisColoringRenderMode, GmVisColorRangeMode
from pcfitting import data_loading, GMSampler
import matplotlib
import matplotlib.image as mimg
matplotlib.use('TkAgg')

gmm_path = r"K:\DA-Eval\dataset_eval_big\gmms"
rendering_path = r"K:\DA-Eval\dataset_eval_big\renderings"

# better only select one of them
RECPC = False

vis = GMVisualizer(False, 800, 800)
vis.set_camera_auto(True)
vis.set_density_rendering(True)
vis.set_ellipsoids_pc_rendering(False, RECPC, False)
vis.set_whitemode(True)

for root, dirs, files in os.walk(gmm_path):
    for name in files:
        if name.lower().endswith(".gma.ply"):
            path = os.path.join(root, name)
            relpath = path[len(gmm_path) + 1:]
            print(relpath)
            gm = data_loading.read_gm_from_ply(path, False)
            vis.set_gaussian_mixtures(gm.cpu())
            if RECPC:
                recpc = GMSampler.sampleGM_ext(gm, 100000)
                vis.set_pointclouds(recpc.view(1, 100000, 3).cpu())
            res = vis.render()
            if RECPC:
                mimg.imsave(os.path.join(rendering_path, "density-" + name[:-8] + ".png"), res[0, 1])
                mimg.imsave(os.path.join(rendering_path, "recpc-" + name[:-8] + ".png"), res[0, 0])
            else:
                mimg.imsave(os.path.join(rendering_path, "density-" + name[:-8] + ".png"), res[0, 0])
print("Done")