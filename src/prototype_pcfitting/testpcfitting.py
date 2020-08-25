from prototype_pcfitting import PCDatasetIterator, Scaler, GMLogger
from prototype_pcfitting.generators import GradientDescentGenerator
import datetime

# model_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/ModelNet10/ModelNet10"
# genpc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/ModelNet10/genpc"
# gengmm_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmmoutput"

model_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/models"
genpc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/pointclouds"
gengmm_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/gmms"
log_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/logs"

log_prefix = input('Name for this training (or empty for auto): ')
if log_prefix == '':
    log_prefix = f'fitPointcloud_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

n_points = 20000
n_gaussians = 100

dataset = PCDatasetIterator(model_path, n_points, 20, genpc_path)
scaler = Scaler()
generator = GradientDescentGenerator(n_components=n_gaussians, n_sample_points=1000)

i = 1
while dataset.has_next():
    print("-----------------------------------------------------------------------------------------")
    batch, names = dataset.next_batch()
    print(f"Dataset Batch {i}: {batch.shape}, Remaining Batches: {dataset.remaining_batches_count()}")

    scaler.set_pointcloud_batch(batch)
    batch_scaled = scaler.scale_down_pc(batch)

    logger = GMLogger(names=names, log_prefix=log_prefix, log_path=log_path, log_positions=500,
                      gm_n_components=n_gaussians, log_loss_console=1, log_loss_tb=1, log_rendering_tb=250, log_gm=250,
                      pointclouds=batch, scaler=scaler)

    generator.set_logging(logger)
    generator.generate(batch_scaled)

    i += 1

print("Done")
