import os
import sqlite3
from math import sqrt

from pcfitting.eval_scripts.eval_db_access_v2 import EvalDbAccessV2
from pcfitting import programs, MaxIterationTerminationCriterion, RelChangeTerminationCriterion, PCDatasetIterator, \
    data_loading, GMSampler
from pcfitting.generators import GradientDescentGenerator, EMGenerator, EckartGeneratorSP, EckartGeneratorHP, PreinerGenerator
from pcfitting.error_functions import AvgDensities, ReconstructionStats, GMMStats
import time
import gmc.mixture
from gmc.cpp.gm_vis.gm_vis import GMVisualizer
import torch
import matplotlib
import matplotlib.image as mimg
import shutil
matplotlib.use('TkAgg')

rendering_path = r"K:\DA-Eval\dataset_eval_big\renderings"
goal_path = r"K:\DA-Eval\dataset_eval_big\renderings\orderings"
db_path = r"K:\DA-Eval\EvalV3.db"

table_of_interest = "EvalDistance"
metric_of_interest = "std_s_projfil"
model_of_interest = "bed_0001.off"

save_path = os.path.join(goal_path, model_of_interest, metric_of_interest)
if not os.path.exists(save_path):
    os.makedirs(save_path)

dbaccess = EvalDbAccessV2(db_path)

cur = dbaccess.connection().cursor()
sql = "SELECT Run.ID, " + metric_of_interest + " FROM " + table_of_interest + " JOIN Run ON " + table_of_interest + ".run = Run.ID WHERE Run.modelfile = ? ORDER BY " + metric_of_interest
cur.execute(sql, (model_of_interest,))
evals = cur.fetchall()

i = -1
for data in evals:
    i += 1
    runid = data[0]
    eval = data[1]
    print(i, " / ", (100 * i / len(evals)), "%")
    density_from = os.path.join(rendering_path, "density-" + str(runid).zfill(9) + ".png")
    density_to = os.path.join(save_path, "density-" + str(i).zfill(5) + "-" + str(runid).zfill(9) + "(" + str(eval) + ")" + ".png")
    recpc_from = os.path.join(rendering_path, "recpc-" + str(runid).zfill(9) + ".png")
    shutil.copyfile(density_from, density_to)
    if os.path.exists(recpc_from):
        recpc_to = os.path.join(save_path, "recpc-" + str(i).zfill(5) + "-" + str(runid).zfill(9) + "(" + str(eval) + ")" + ".png")
        shutil.copyfile(recpc_from, recpc_to)

