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
matplotlib.use('TkAgg')

db_path = r"F:\DA-Eval\EvalV2.db"

evalstats = GMMStats()
dbaccess = EvalDbAccessV2(db_path)

cur = dbaccess.connection().cursor()
sql = "SELECT * FROM EvalDistance"
cur.execute(sql)
evals = cur.fetchall()

i = -1
for data in evals:
    evid = data[0]
    rmsds = data[1]
    rmsdg = data[5]
    rcd = sqrt(rmsds**2 + rmsdg**2)
    print(evid, " / ", (100 * evid / len(evals)), "%")

    sql = "UPDATE EvalDistance SET rcd = ? WHERE ID = ?"
    cur = dbaccess.connection().cursor()
    cur.execute(sql, (rcd, evid))
    dbaccess.connection().commit()

