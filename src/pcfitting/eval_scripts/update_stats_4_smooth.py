import os
import sqlite3

from pcfitting.eval_scripts.eval_db_access_v2 import EvalDbAccessV2
from pcfitting import programs, MaxIterationTerminationCriterion, RelChangeTerminationCriterion, PCDatasetIterator, \
    data_loading, GMSampler
from pcfitting.generators import GradientDescentGenerator, EMGenerator, EckartGeneratorSP, EckartGeneratorHP, PreinerGenerator
from pcfitting.error_functions import AvgDensities, ReconstructionStats, GMMStats, SmoothnessOfDensity
import time
import gmc.mixture
from gmc.cpp.gm_vis.gm_vis import GMVisualizer
import torch
import matplotlib
import matplotlib.image as mimg
matplotlib.use('TkAgg')

model_path = r"K:\DA-Eval\dataset_eval_big\models"
fitpc_path = r"K:\DA-Eval\dataset_eval_big\fitpcs"
evalpc_path = r"K:\DA-Eval\dataset_eval_big\evalpcs\n1000000"
recpc_path = r"K:\DA-Eval\dataset_eval_big\recpcs"
gengmm_path = r"K:\DA-Eval\dataset_eval_big\gmms"
rendering_path = r"K:\DA-Eval\dataset_eval_big\renderings"
db_path = r"K:\DA-Eval\EvalV3.db"

#evalstats = GMMStats(False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True)
evalsmooth = SmoothnessOfDensity(subsamples=10000)
dbaccess = EvalDbAccessV2(db_path)

cur = dbaccess.connection().cursor()
sql = "SELECT EvalDensity.ID, EvalDensity.run, Run.modelfile FROM EvalDensity JOIN Run ON EvalDensity.run = Run.id"
cur.execute(sql)
stats = cur.fetchall()

i = -1
for stat in stats:
    eid = stat[0]
    runid = stat[1]
    modelfile = stat[2]
    print(runid, " / ", (100 * eid / len(stats)), "%")
    gma = data_loading.read_gm_from_ply(os.path.join(gengmm_path, str(runid).zfill(9) + ".gma.ply"), ismodel=False)
    pcpath = os.path.join(evalpc_path, modelfile)
    pcbatch = data_loading.load_pc_from_off(pcpath)
    print(" Evaluating")
    modelpath = os.path.join(model_path, modelfile)
    statvalues = evalsmooth.calculate_score_packed(pcbatch, gma, modelpath=modelpath)
    print(" Saving")

    sql = "UPDATE EvalDensity SET smooth = ? WHERE ID = ?"
    dbaccess.connection().cursor().execute(sql, (statvalues[0].item(), eid))
    dbaccess.connection().commit()

