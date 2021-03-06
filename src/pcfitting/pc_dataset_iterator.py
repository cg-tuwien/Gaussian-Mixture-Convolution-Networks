import os
import torch
from queue import SimpleQueue
from pcfitting import data_loading
import trimesh
import trimesh.sample
import math


class DatasetIterator:
    # The DatasetIterator is used for iterating over a model dataset
    # It reads 3d models from a given directory and samples them into
    # a point cloud of a given point count. The point clouds are created
    # and returned in batches. They are also stored in a directory and
    # loaded from there if the model directory has been used before.

    def has_next(self) -> bool:
        # Returns true if more batches are available
        pass

    def next_batch(self):
        # returns a tensor of the size [b,n,3], where b is the batch size (or less, if less data was available)
        # and n is the point count
        # also returns a list of names of the point clouds
        pass

    def remaining_batches_count(self):
        pass


class PCDatasetIterator(DatasetIterator):
    # The PPCDatasetIterator is used for iterating over a model dataset
    # It reads 3d models from a given directory and samples them into
    # a point cloud of a given point count. The point clouds are created
    # and returned in batches. They are also stored in a directory and
    # loaded from there if the model directory has been used before.

    def __init__(self, batch_size: int, point_count: int, pc_root: str, model_root: str = None):
        # Constructor
        # Creates a new PCDatasetIterator. If model_root is given, the point clouds will be sampled
        # from the models and stored in the folder {pc_root}/n{point_count}". If not, the point clouds
        # will simply be read from pc_root and it's subdirectories.
        # Parameters:
        #   batch_size: int
        #       How many point clouds will be returned in one batch
        #   point_count: int
        #       With how many points point clouds the pointclouds (will) have
        #   pc_root: str
        #       The path to either read the point clouds from or store the point clouds in.
        #   model_root: str
        #       Directory containing the models to load. Subdirectories are checked too!
        #
        self._file_queue = SimpleQueue()
        self._model_root = model_root
        self._point_count = point_count
        self._batch_size = batch_size
        if self._model_root is None:
            self._pc_root = pc_root
        else:
            self._pc_root = os.path.join(pc_root, "n" + str(point_count))
        for root, dirs, files in os.walk(model_root if model_root is not None else pc_root):
            for name in files:
                if name.lower().endswith(".off"):
                    path = os.path.join(root, name)
                    relpath = path[len(model_root if model_root is not None else pc_root) + 1:]
                    self._file_queue.put(relpath)

    def has_next(self) -> bool:
        # Returns true if more batches are available
        return not self._file_queue.empty()

    def skip_batch(self):
        self._file_queue.get()

    def next_batch(self):
        # returns a tensor of the size [b,n,3], where b is the batch size (or less, if less data was available)
        # and n is the point count
        # also returns a list of names of the point clouds
        current_batch_size = min(self._batch_size, self._file_queue.qsize())
        batch = torch.zeros(current_batch_size, self._point_count, 3, device=torch.device("cuda"))
        names = [None] * current_batch_size
        for i in range(self._batch_size):
            if not self._file_queue.empty():
                filename = self._file_queue.get()
                names[i] = filename
                if self._model_root is not None:
                    objpath = os.path.join(self._model_root, filename)
                    pcpath = os.path.join(self._pc_root, filename)
                    if os.path.exists(pcpath):
                        batch[i, :, :] = data_loading.load_pc_from_off(pcpath)[0, :, :]
                    else:
                        print("Sampling ", objpath)
                        mesh = trimesh.load_mesh(objpath)
                        samples, _ = trimesh.sample.sample_surface(mesh, self._point_count)
                        data_loading.write_pc_to_off(pcpath, samples)
                        batch[i, :, :] = torch.from_numpy(samples)
                else:
                    pcpath = os.path.join(self._pc_root, filename)
                    loaded = data_loading.load_pc_from_off(pcpath)
                    if loaded.shape[1] != self._point_count:
                        raise Exception("There are point clouds with different point count in the given directory!")
                    batch[i, :, :] = loaded[0, :, :]
        return batch, names

    def remaining_batches_count(self):
        return math.ceil(self._file_queue.qsize() / self._batch_size)
