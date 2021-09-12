import os
import sqlite3

from pcfitting.eval_scripts.eval_db_access_v2 import EvalDbAccessV2
from pcfitting import programs, MaxIterationTerminationCriterion, RelChangeTerminationCriterion, PCDatasetIterator, \
    data_loading, GMSampler
from pcfitting.generators import GradientDescentGenerator, EMGenerator, EckartGeneratorSP, EckartGeneratorHP, PreinerGenerator
from pcfitting.error_functions import AvgDensities, ReconstructionStats, GMMStats, Irregularity
import time
import gmc.mixture
from gmc.cpp.gm_vis.gm_vis import GMVisualizer
import torch
import matplotlib
import matplotlib.image as mimg
matplotlib.use('TkAgg')

model_path = r"K:\DA-Eval\dataset_eval_big\models"
fitpc_path = r"K:\DA-Eval\dataset_eval_big\fitpcs\n100000"
evalpc_path = r"K:\DA-Eval\dataset_eval_big\evalpcs\n1000000"
recpc_path = r"K:\DA-Eval\dataset_eval_big\recpcs"
gengmm_path = r"K:\DA-Eval\dataset_eval_big\gmms"
rendering_path = r"K:\DA-Eval\dataset_eval_big\renderings"
db_path = r"K:\DA-Eval\EvalV3.db"

#evalstats = GMMStats(False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True)
dbaccess = EvalDbAccessV2(db_path)

cur = dbaccess.connection().cursor()
sql = "SELECT Run.ID, OptionsEM.init_method, EvalStats.ID, Run.modelfile FROM Run JOIN OptionsEM ON Run.method_options = OptionsEM.ID AND Run.method = 'EM' Join EvalStats ON EvalStats.run = Run.ID WHERE OptionsEM.eps = 1e-5 AND Run.nr_fit_points = 100000 AND Run.n_gaussians_should = 512 AND OptionsEM.termination_criterion <> 'MaxIter(0)'"
cur.execute(sql)
stats = cur.fetchall()

terminator2 = RelChangeTerminationCriterion(0.1, 20)
i = -1
for stat in stats:
    i = i + 1
    runid = stat[0]
    init = stat[1]
    eid = stat[2]
    modelfile = stat[3]
    print(runid, " / ", (100 * i / len(stats)), "%")
    pcpath = os.path.join(fitpc_path, modelfile)
    pcbatch = data_loading.load_pc_from_off(pcpath)
    generator = EMGenerator(n_gaussians=512, initialization_method=init, termination_criterion=terminator2, em_step_points_subbatchsize=10000, verbosity=0, eps=1e-5)
    print("Generating")
    generator.generate(pcbatch)
    it = generator.final_nr_iterations
    print("Saving")

    sql = "UPDATE EvalStats SET nr_iterations = ? WHERE ID = ?"
    dbaccess.connection().cursor().execute(sql, (it, eid))
    dbaccess.connection().commit()

