import os
from gmc.cpp.gm_vis.gm_vis import GMVisualizer
from pcfitting import data_loading
import matplotlib
import matplotlib.image as mimg
matplotlib.use('TkAgg')

gmm_path = r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\thesis\EvalDraft\images\gmms"
rendering_path = r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\thesis\EvalDraft\images\renderings"

vis = GMVisualizer(False, 800, 800)
vis.set_camera_auto(True)
vis.set_density_rendering(True)
vis.set_ellipsoids_pc_rendering(False, False, False)
vis.set_camera_lookat(( 73.4169 , 79.4967 , 67.152), (  -0.339551 , -6.20043 , 9.12231), (0, 1, 0))

for root, dirs, files in os.walk(gmm_path):
    for name in files:
        if name.lower().endswith(".gma.ply"):
            path = os.path.join(root, name)
            relpath = path[len(gmm_path) + 1:]
            print(relpath)
            gm = data_loading.read_gm_from_ply(path, False)
            vis.set_whitemode(False)
            vis.set_gaussian_mixtures(gm.cpu())
            res = vis.render()
            mimg.imsave(os.path.join(rendering_path, "density-" + name[:-8] + "-b.png"), res[0, 0])
            vis.set_whitemode(True)
            res = vis.render()
            mimg.imsave(os.path.join(rendering_path, "density-" + name[:-8] + "-w.png"), res[0, 0])
print("Done")