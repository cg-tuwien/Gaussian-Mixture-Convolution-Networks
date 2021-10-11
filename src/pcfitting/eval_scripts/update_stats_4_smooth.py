import os
import sqlite3
import time

from pcfitting.eval_scripts.eval_db_access_v2 import EvalDbAccessV2
from pcfitting import programs, MaxIterationTerminationCriterion, RelChangeTerminationCriterion, PCDatasetIterator, \
    data_loading, GMSampler
from pcfitting.generators import GradientDescentGenerator, EMGenerator, EckartGeneratorSP, EckartGeneratorHP, PreinerGenerator
from pcfitting.error_functions import AvgDensities, ReconstructionStats, GMMStats, Irregularity, ReconstructionStatsProjected, ReconstructionStatsFiltered
import time
import gmc.mixture
from gmc.cpp.gm_vis.gm_vis import GMVisualizer
import torch
import matplotlib
import matplotlib.image as mimg
matplotlib.use('TkAgg')



model_path = r"K:\DA-Eval\dataset_eval_big\models"
fitpc_path = r"K:\DA-Eval\dataset_eval_big\fitpcs"
evalpc_path = r"K:\DA-Eval\dataset_eval_big\evalpcs\n100000"
recpc_path = r"K:\DA-Eval\dataset_eval_big\recpcs"
gengmm_path = r"K:\DA-Eval\dataset_eval_big\gmms"
rendering_path = r"K:\DA-Eval\dataset_eval_big\renderings"
db_path = r"K:\DA-Eval\EvalV3.db"

#evalstats = GMMStats(False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True)
#evalsmooth = Irregularity(subsamples=10000)
recstats = ReconstructionStatsFiltered(ReconstructionStatsProjected(ReconstructionStats(rmsd_scaled_by_nn=False, md_scaled_by_nn=False, cv=True, inverse=False, chamfer_norm_nn=False, sample_points=100000)))
dbaccess = EvalDbAccessV2(db_path)

cur = dbaccess.connection().cursor()
sql = "SELECT EvalDistance.ID, EvalDistance.run, Run.modelfile, NNScaling.factor FROM EvalDistance JOIN Run ON EvalDistance.run = Run.id JOIN NNScaling ON Run.modelfile = NNScaling.modelfile WHERE EvalDistance.std_s_projfil IS NULL"
cur.execute(sql)
stats = cur.fetchall()

i = 0
for stat in stats:
    i = i + 1
    eid = stat[0]
    runid = stat[1]
    modelfile = stat[2]
    nnfactor = stat[3]
    print(runid, " / ", (100 * i / len(stats)), "%")
    gma = data_loading.read_gm_from_ply(os.path.join(gengmm_path, str(runid).zfill(9) + ".gma.ply"), ismodel=False)
    pcpath = os.path.join(evalpc_path, modelfile)
    pcbatch = data_loading.load_pc_from_off(pcpath)
    pcbatch.nnscalefactor = nnfactor
    print(" Evaluating")
    modelpath = os.path.join(model_path, modelfile)
    statvalues = recstats.calculate_score_packed(pcbatch, gma, modelpath=modelpath)
    print(" Saving")
    sql = "UPDATE EvalDistance SET std_s_projfil = ?, cv_s_projfil = ? WHERE ID = ?"
    dbaccess.connection().cursor().execute(sql, (statvalues[0].item(), statvalues[1].item(), eid))
    dbaccess.connection().commit()

