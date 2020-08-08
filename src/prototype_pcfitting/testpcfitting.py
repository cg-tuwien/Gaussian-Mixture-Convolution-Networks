from prototype_pcfitting.pc_dataset_iterator import PCDatasetIterator

# model_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/ModelNet10/ModelNet10"
# genpc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/ModelNet10/genpc"
# gengmm_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmmoutput"

model_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/models"
genpc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/pointclouds"
gengmm_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/gmms"

dataset = PCDatasetIterator(model_path, 500, 20, genpc_path)

i = 1
while dataset.has_next():
    batch = dataset.next_batch()
    print(f"Dataset Batch {i}: {batch.shape}, Remaining Batches: {dataset.remaining_batches_count()}")
    i += 1

print("Done")
