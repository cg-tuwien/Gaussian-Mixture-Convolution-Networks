# ------------------------------------------------
# Useful script for precalculating the nn-scaling-factors for evaluation
# Can be deleted if not needed anymore
# ------------------------------------------------

from pcfitting.cpp.gmeval import pyeval
from pcfitting import PCDatasetIterator
from pcfitting.eval_scripts.eval_db_access_v2 import EvalDbAccessV2
import math

model_path = r"K:\DA-Eval\dataset_eval_big\models"
evalpc_path = r"K:\DA-Eval\dataset_eval_big\evalpcs"
db_path = r"K:\DA-Eval\EvalV3.db"

dataset_eval_dist = PCDatasetIterator(1, 100000, evalpc_path, model_path)
dbaccess = EvalDbAccessV2(db_path)

print("Start")
while dataset_eval_dist.has_next():
    pcbatch, names = dataset_eval_dist.next_batch()
    md = pyeval.calc_rmsd_to_itself(pcbatch.view(-1, 3))[1]
    refdist = 128 / (2*math.sqrt(pcbatch.shape[1]) - 1)
    scalefactor = refdist / md
    print(names, ": ", scalefactor)
    dbaccess.save_nn_scale_factor(names[0], scalefactor)
    # assert pcbatch.shape[1] == 100000
    # print(names)


# ['airplane_0001.off'] :  0.3636986333108109
# Sampling  F:\DA-Eval\dataset_eval\models\bathtub_0001.off
# ['bathtub_0001.off'] :  1.0074831411725227
# Sampling  F:\DA-Eval\dataset_eval\models\bed_0001.off
# ['bed_0001.off'] :  0.8185107750410209
# Sampling  F:\DA-Eval\dataset_eval\models\bench_0001.off
# ['bench_0001.off'] :  0.5590135529590764
# Sampling  F:\DA-Eval\dataset_eval\models\bookshelf_0001.off
# ['bookshelf_0001.off'] :  1.4423186003159514
# Sampling  F:\DA-Eval\dataset_eval\models\bowl_0001.off
# ['bowl_0001.off'] :  3.960926161827602
# Sampling  F:\DA-Eval\dataset_eval\models\car_0001.off
# ['car_0001.off'] :  0.10709724848967923
# Sampling  F:\DA-Eval\dataset_eval\models\chair_0001.off
# ['chair_0001.off'] :  4.468706756010175
# Sampling  F:\DA-Eval\dataset_eval\models\curtain_0001.off
# ['curtain_0001.off'] :  0.1436627389788331
# Sampling  F:\DA-Eval\dataset_eval\models\desk_0001.off
# ['desk_0001.off'] :  0.7635226298467559
# Sampling  F:\DA-Eval\dataset_eval\models\guitar_0001.off
# ['guitar_0001.off'] :  1.432110143135819
# Sampling  F:\DA-Eval\dataset_eval\models\monitor_0001.off
# ['monitor_0001.off'] :  4.438397248958837
# Sampling  F:\DA-Eval\dataset_eval\models\person_0001.off
# ['person_0001.off'] :  2.568091849895514
# Sampling  F:\DA-Eval\dataset_eval\models\piano_0001.off
# ['piano_0001.off'] :  0.9321083621463199
# Sampling  F:\DA-Eval\dataset_eval\models\plant_0001.off
# ['plant_0001.off'] :  3.372988219485817
# Sampling  F:\DA-Eval\dataset_eval\models\stool_0001.off
# ['stool_0001.off'] :  1.658202550953507
# Sampling  F:\DA-Eval\dataset_eval\models\table_0001.off
# ['table_0001.off'] :  1.5401110297747662
# Sampling  F:\DA-Eval\dataset_eval\models\tent_0001.off
# ['tent_0001.off'] :  0.714706594397241
# Sampling  F:\DA-Eval\dataset_eval\models\toilet_0001.off
# ['toilet_0001.off'] :  2.3047958525099
# Sampling  F:\DA-Eval\dataset_eval\models\vase_0001.off
# ['vase_0001.off'] :  6.055677648526986
