from pcfitting import PCDatasetIterator

# Samples models two times differently with the same point count and stores the results

model_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/models"
pc1_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vt_evaluation/fitpcs"
pc2_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vt_evaluation/evalpcs"

pointcount = 100000

pcdi1 = PCDatasetIterator(1, pointcount, pc1_path, model_path)
pcdi2 = PCDatasetIterator(1, pointcount, pc2_path, model_path)

while pcdi1.has_next():
    pcdi1.next_batch()
    pcdi2.next_batch()