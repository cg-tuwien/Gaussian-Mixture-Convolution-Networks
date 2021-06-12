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

dbaccess = EvalDbAccessV2(db_path)

cur = dbaccess.connection().cursor()
sql = "SELECT * FROM EvalStats"
cur.execute(sql)
stats = cur.fetchall()

i = -1
for stat in stats:
    id = stat[0]
    cv = stat[2] / stat[1]
    print(id, " / ", (100 * id / len(stats)), "%", cv)
    sql = "UPDATE EvalStats SET cv_traces = ? WHERE ID = ?"
    dbaccess.connection().cursor().execute(sql, (cv, id))
    dbaccess.connection().commit()

