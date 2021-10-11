# ------------------------------------------------
# Useful script for creating the visualizations for a folder of GMMs
# Can be deleted if not needed anymore
# ------------------------------------------------
import os
from gmc.cpp.gm_vis.gm_vis import GMVisualizer, GmVisColoringRenderMode, GmVisColorRangeMode
from pcfitting import data_loading, GMSampler
from pcfitting.eval_scripts.eval_db_access_v2 import EvalDbAccessV2
import matplotlib
import matplotlib.image as mimg
matplotlib.use('TkAgg')

gmm_path = r"K:\DA-Eval\dataset_eval_big\gmms"
rendering_path = r"K:\DA-Eval\dataset_eval_big\renderings"
db_path = r"K:\DA-Eval\EvalV3.db"

model_of_interest = "laptop_0002.off"

dbaccess = EvalDbAccessV2(db_path)

cur = dbaccess.connection().cursor()
sql = "SELECT Run.ID FROM Run WHERE Run.modelfile = ?"
cur.execute(sql, (model_of_interest,))
evals = cur.fetchall()

# better only select one of them
ELLIP = False
RECPC = True

vis = GMVisualizer(False, 800, 800)
vis.set_camera_auto(True)
vis.set_density_rendering(True)
vis.set_whitemode(True)
vis.set_ellipsoids_pc_rendering(ELLIP, RECPC, ELLIP)
vis.set_ellipsoids_colormode(GmVisColoringRenderMode.COLOR_WEIGHT)
#vis.set_ellipsoids_rangemode(GmVisColorRangeMode.RANGE_MINMAX)
vis.set_ellipsoids_rangemode(GmVisColorRangeMode.RANGE_MANUAL, 0, 0.01)
vis.set_density_rendering(False)
#vis.set_camera_lookat(( 73.4169 , 79.4967 , 67.152), (  -0.339551 , -6.20043 , 9.12231), (0, 1, 0))

for res in evals:
    name = str(res[0]).zfill(9) + ".gma.ply"
    path = os.path.join(gmm_path, name)
    relpath = path[len(gmm_path) + 1:]
    print(relpath)
    gm = data_loading.read_gm_from_ply(path, False)
    vis.set_gaussian_mixtures(gm.cpu())
    if RECPC:
        recpc = GMSampler.sampleGM_ext(gm, 100000)
        vis.set_pointclouds(recpc.view(1, 100000, 3).cpu())
    res = vis.render()
    if ELLIP:
        mimg.imsave(os.path.join(rendering_path, "ellip-" + name[:-8] + ".png"), res[0, 0])
    elif RECPC:
        mimg.imsave(os.path.join(rendering_path, "recpc-" + name[:-8] + ".png"), res[0, 0])
print("Done")