from pcfitting.eval_db_access import EvalDbAccess

access = EvalDbAccess(r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\EvalLogs\Eval01.db")

print ("OptionsEM: ", access.insert_options_em(100000, "MaxIter(200)", "fpsmax", "float32", 1e-7, True))
print ("OptionsEckHP: ", access.insert_options_eckart_hp(8, 4, "MaxIter(200)", "fpsmax", "float32", 1e-7, True))
print ("OptionsEckSP: ", access.insert_options_eckart_sp(8, 4, 0.2, "RelChange(...)", "Eigen", "float64", 0.003, False))
preiner = access.insert_options_preiner(2.0, True, 1.0, False, "fixed", 0, 0.5, False, 3, 0.3, 512, True)
print ("OptionsPreiner: ", preiner)
run = access.insert_run("modelfile", 100000, 1000000, 512, 511, "Preiner", preiner, 1.24)
print("Run: ", run)
print ("DistanceEval: ", access.insert_distance_eval(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,run,True))
print ("DensityEval: ", access.insert_density_eval(101,102,103,104,105,106,107,108,None,run,True))
print ("StatEval: ", access.insert_stat_eval(51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 0.9, 1, 2, run))