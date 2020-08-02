from prototype_pcfitting.pcdatasetiterator import PCDatasetIterator

model_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/ModelNet10/ModelNet10"
genpc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/ModelNet10/genpc"
gengmm_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmmoutput"

dataset = PCDatasetIterator(model_path, 500, 20, genpc_path)