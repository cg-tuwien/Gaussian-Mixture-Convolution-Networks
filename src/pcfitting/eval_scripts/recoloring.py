# ------------------------------------------------
# Useful script for creating the visualizations for a folder of GMMs
# Can be deleted if not needed anymore
# ------------------------------------------------
import os
from gmc.cpp.gm_vis.gm_vis import GMVisualizer, GmVisColoringRenderMode, GmVisColorRangeMode
from pcfitting import data_loading, GMSampler
import matplotlib
import matplotlib.image as mimg
matplotlib.use('TkAgg')

gmm_path = r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\data\dataset_plane\gmms\211006-x1-EckInitsFixed"
# gmm_path = r"K:\DA-Eval\dataset_eval_big\recpcs"
# gmm_path = r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\thesis\graphics\renderings"
rendering_path = r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\data\dataset_plane\renderings\eckinitsfixed-rot"
# rendering_path = r"K:\DA-Eval\dataset_eval_big\recpcs"
# rendering_path = r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\thesis\graphics\renderings\new"

# better only select one of them
ELLIP = True
RECPC = False

vis = GMVisualizer(False, 800, 800)
#vis.set_camera_auto(True)
vis.set_density_rendering(True)
vis.set_ellipsoids_pc_rendering(ELLIP, RECPC, ELLIP)
vis.set_ellipsoids_colormode(GmVisColoringRenderMode.COLOR_WEIGHT)
vis.set_ellipsoids_rangemode(GmVisColorRangeMode.RANGE_MINMAX)
#vis.set_ellipsoids_rangemode(GmVisColorRangeMode.RANGE_MANUAL, 0, 0.01)
#vis.set_camera_lookat(( 0, 0, 16.5), ( 0, 0, 0), (0, 1, 0))
vis.set_camera_lookat(( 3.2337, 0, 15.6698), ( 0, 0, 0), (0, 1, 0))

for root, dirs, files in os.walk(gmm_path):
    for name in files:
        if name.lower().endswith(".gma.ply"):
        #if name.lower().endswith(".gma.ply"):
            path = os.path.join(root, name)
            relpath = path[len(gmm_path) + 1:]
            print(relpath)
            gm = data_loading.read_gm_from_ply(path, False)
            vis.set_whitemode(False)
            vis.set_gaussian_mixtures(gm.cpu())
            if RECPC:
                recpc = GMSampler.sampleGM_ext(gm, 100000)
                vis.set_pointclouds(recpc.view(1, 100000, 3).cpu())
            res = vis.render() # relpath.replace("\\","-") +
            mimg.imsave(os.path.join(rendering_path, "density-" + relpath.replace("\\","-") +name[:-8] + "-b.png"), res[0, 1])
            if ELLIP:
                mimg.imsave(os.path.join(rendering_path, "ellip-" + relpath.replace("\\","-") +name[:-8] + "-b.png"), res[0, 0])
            elif RECPC:
                mimg.imsave(os.path.join(rendering_path, "recpc-" + relpath.replace("\\","-") +name[:-8] + "-b.png"), res[0, 0])
            vis.set_whitemode(True)
            res = vis.render()
            mimg.imsave(os.path.join(rendering_path, "density-" + relpath.replace("\\","-") +name[:-8] + "-w.png"), res[0, 1])
            if ELLIP:
                mimg.imsave(os.path.join(rendering_path, "ellip-" + relpath.replace("\\","-") +name[:-8] + "-w.png"), res[0, 0])
            elif RECPC:
                mimg.imsave(os.path.join(rendering_path, "recpc-" + relpath.replace("\\","-") +name[:-8] + "-w.png"), res[0, 0])
print("Done")