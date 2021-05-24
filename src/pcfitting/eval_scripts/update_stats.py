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

model_path = r"F:\DA-Eval\dataset_eval\models"
fitpc_path = r"F:\DA-Eval\dataset_eval\fitpcs"
evalpc_path = r"F:\DA-Eval\dataset_eval\evalpcs"
recpc_path = r"F:\DA-Eval\dataset_eval\recpcs"
gengmm_path = r"F:\DA-Eval\dataset_eval\gmms"
rendering_path = r"F:\DA-Eval\dataset_eval\renderings"
db_path = r"F:\DA-Eval\EvalV2.db"

evalstats = GMMStats()
dbaccess = EvalDbAccessV2(db_path)

cur = dbaccess.connection().cursor()
sql = "SELECT * FROM RUN"
cur.execute(sql)
runs = cur.fetchall()

i = -1
for run in runs:
    runid = run[0]
    modelfile = run[1]
    print(runid, " / ", (100 * runid / len(runs)), "%")
    gma = data_loading.read_gm_from_ply(os.path.join(gengmm_path, str(runid).zfill(9) + ".gma.ply"), ismodel=False)
    nns = dbaccess.get_nn_scale_factor(modelfile)
    dummypcbatch = torch.zeros(1)
    dummypcbatch.nnscalefactor = nns
    print(" Evaluating")
    statvalues = evalstats.calculate_score_packed(dummypcbatch, gma)
    print(" Saving")

    dbaccess.insert_eval_stat(statvalues[0].item(), statvalues[1].item(), statvalues[2].item(),
                              statvalues[3].item(), statvalues[4].item(), statvalues[5].item(),
                              statvalues[6].item(), statvalues[7].item(), statvalues[8].item(),
                              statvalues[9].item(), statvalues[10].item(), statvalues[11].item(),
                              statvalues[12].item(), statvalues[13].item(), statvalues[14].item(),
                              statvalues[15].item(), statvalues[16].item(), statvalues[17].item(), runid)

