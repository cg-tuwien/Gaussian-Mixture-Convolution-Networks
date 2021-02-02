import os
import torch
from queue import SimpleQueue
from prototype_pcfitting import data_loading
import math
import numpy as np
import prototype_pcfitting.pc_dataset_iterator


def load_points_from_file_to_numpy(filename,
                                   delimiter=',',
                                   dimension=3,
                                   max_num_points=None,
                                   dtype=np.float32):
    points = []
    features = []
    with open(filename) as in_file:
        for i, line in enumerate(in_file):
            if max_num_points is None or i < max_num_points:
                line_elements = line[:-1].split(delimiter)
                points.append(line_elements[0:dimension])
                if len(line_elements) > 3:
                    features.append(line_elements[dimension:])
            else:
                break

        points = torch.from_numpy(np.array(points, dtype=dtype))
        features = torch.from_numpy(np.array(features, dtype=dtype))
        return points, features
    assert False


class ModelNetDatasetIterator(prototype_pcfitting.pc_dataset_iterator.DatasetIterator):
    # The PPCDatasetIterator is used for iterating over a model dataset
    # It reads 3d models from a given directory and samples them into
    # a point cloud of a given point count. The point clouds are created
    # and returned in batches. They are also stored in a directory and
    # loaded from there if the model directory has been used before.

    def __init__(self, batch_size: int, dataset_path: str = None, filelist_name: str = 'filelist.txt'):
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
        self._batch_size = batch_size
        self._file_queue = SimpleQueue()
        self._dataset_path = dataset_path
        file_name = None
        with open(f"{dataset_path}/{filelist_name}") as inFile:
            for line in inFile:
                l = line.replace('\n', '')
                file_name = f"{l}"
                self._file_queue.put(file_name)
            assert self._file_queue.qsize() > 0
        assert file_name is not None
        points, _ = load_points_from_file_to_numpy(f"{self._dataset_path}/{file_name}")
        self._point_count = points.shape[0]

    def has_next(self) -> bool:
        # Returns true if more batches are available
        return not self._file_queue.empty()

    def next_batch(self):
        # returns a tensor of the size [b,n,3], where b is the batch size (or less, if less data was available)
        # and n is the point count
        # also returns a list of names of the point clouds
        current_batch_size = min(self._batch_size, self._file_queue.qsize())

        batch = torch.zeros(current_batch_size, self._point_count, 3, device=torch.device("cuda"))
        names = [None] * current_batch_size
        for i in range(current_batch_size):
            assert(not self._file_queue.empty())
            filename = self._file_queue.get()
            names[i] = filename.replace('.txt', '')
            points, features = _ = load_points_from_file_to_numpy(filename=f"{self._dataset_path}/{filename}")
            assert(points.shape[0] == self._point_count)
            batch[i, :, :] = points

        return batch, names

    def remaining_batches_count(self):
        return math.ceil(self._file_queue.qsize() / self._batch_size)
