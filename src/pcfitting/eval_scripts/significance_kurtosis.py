from pcfitting import MaxIterationTerminationCriterion, RelChangeTerminationCriterion, PCDatasetIterator, GMSampler, data_loading
from pcfitting.generators import EMGenerator, EckartGeneratorSP, PreinerGenerator
from pcfitting.error_functions import AvgDensities, ReconstructionStats, ReconstructionStatsProjected, GMMStats
from gmc.cpp.gm_vis.gm_vis import GMVisualizer
import matplotlib
import matplotlib.image as mimg
matplotlib.use('TkAgg')
import gmc.mixture
import sqlite3
import os

model_path = r"F:\DA-Eval\dataset20\models"
fitpc_path = r"F:\DA-Eval\dataset20\fitpcs"
evalpc_path = r"F:\DA-Eval\dataset20\evalpcs"
recpc_path = r"F:\DA-Eval\dataset20\recpcs-significance"
gengmm_path = r"F:\DA-Eval\dataset20\gmms-significance"
rendering_path = r"F:\DA-Eval\dataset20\renderings-significance"
dbpath = r"F:\DA-Eval\Significance.db"

n_fit_points = 100000
n_eval_points_distance = 100000

evalpc_path_full = os.path.join(evalpc_path, "n" + str(n_eval_points_distance))

minid = 88

db = sqlite3.connect(dbpath)
evaldistance = ReconstructionStats()

sql = "SELECT ID, model FROM RUN;"
cur = db.cursor()
cur.execute(sql)
rows = cur.fetchall()

for row in rows:
    id = row[0]
    model = row[1]

    print("ID: ", id)

    if id < minid:
        continue

    pc = data_loading.load_pc_from_off(os.path.join(evalpc_path_full, model))
    gm = data_loading.read_gm_from_ply(os.path.join(gengmm_path, str(id).zfill(9) + str(".gma.ply")), ismodel=False)
    recpc = data_loading.load_pc_from_off(os.path.join(recpc_path, str(id).zfill(9) + str(".off")))

    #distvalues = evaldistance.calculate_score_packed(pc, gm)
    distvalues = evaldistance.calculate_score_on_reconstructed(pc, recpc)

    sql1 = "UPDATE RUN SET KURTOSIS = ?, KURTOSIS_I = ? WHERE ID = ?;"
    cur = db.cursor()
    cur.execute(sql1, (distvalues.kurtosis, distvalues.kurtosisI, id))
    db.commit()

    print(distvalues.kurtosisI)

