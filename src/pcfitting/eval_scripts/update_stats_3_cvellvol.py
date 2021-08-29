import os
import sqlite3

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
matplotlib.use('TkAgg')

model_path = r"K:\DA-Eval\dataset_eval\models"
fitpc_path = r"K:\DA-Eval\dataset_eval\fitpcs"
evalpc_path = r"K:\DA-Eval\dataset_eval\evalpcs"
recpc_path = r"K:\DA-Eval\dataset_eval\recpcs"
gengmm_path = r"K:\DA-Eval\dataset_eval\gmms"
rendering_path = r"K:\DA-Eval\dataset_eval\renderings"
db_path = r"K:\DA-Eval\EvalV2.db"

evalstats = GMMStats(False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True)
dbaccess = EvalDbAccessV2(db_path)

cur = dbaccess.connection().cursor()
sql = "SELECT EvalStats.ID, EvalStats.run, EvalStats.normalized, NNScaling.factor FROM EvalStats JOIN Run ON EvalStats.run = Run.id JOIN NNScaling ON Run.modelfile = NNScaling.modelfile"
cur.execute(sql)
stats = cur.fetchall()

i = -1
for stat in stats:
    eid = stat[0]
    runid = stat[1]
    normalized = stat[2]
    nnfactor = stat[3]
    if normalized == 0: # be aware of how this is set!
        nnfactor = 1
    else:
        continue
    print(runid, " / ", (100 * eid / len(stats)), "%")
    gma = data_loading.read_gm_from_ply(os.path.join(gengmm_path, str(runid).zfill(9) + ".gma.ply"), ismodel=False)
    dummypcbatch = torch.zeros(1)
    dummypcbatch.nnscalefactor = nnfactor
    print(" Evaluating")
    statvalues = evalstats.calculate_score_packed(dummypcbatch, gma)
    print(" Saving")

    sql = "UPDATE EvalStats SET avg_sqrt_det = ?, std_sqrt_det = ?, cv_ellvol = ? WHERE ID = ?"
    dbaccess.connection().cursor().execute(sql, (statvalues[0].item(), statvalues[1].item(), statvalues[2].item(), eid))
    dbaccess.connection().commit()

