import os
import torch
import numpy as np
from queue import SimpleQueue
import prototype_pcfitting.pointcloud as pointcloud
import trimesh
import trimesh.sample

# The PPCDatasetIterator is used for iterating over a model dataset
# It reads 3d models from a given directory and samples them into
# a point cloud of a given point count. The point clouds are created
# and returned in batches. They are also stored in a directory and
# loaded from there if the model directory has been used before.
class PCDatasetIterator:

    # Constructor
    # Creates a new PCDatasetIterator.
    # Parameters:
    #   modelroot: str
    #       Directory containing the models to load. Subdirectories are checked too!
    #   pointcount: int
    #       With how many points point clouds will be sampled from the models
    #   batchsize: int
    #       How many point clouds will be returned in one batch
    #   pcroot: str (optional)
    #       The path to store the point clouds in. If a point cloud for a given model
    #       has also been generated and stored in this directory. It is read from there as well,
    #       instead of creating a new point cloud.
    #
    def __init__(self, modelroot: str, pointcount: int, batchsize: int, pcroot: str = None):
        self.filequeue = SimpleQueue()
        self.modelroot = modelroot
        self.pointcount = pointcount
        self.batchsize = batchsize
        self.pcroot = os.path.join(pcroot,"n"+str(pointcount))
        for root, dirs, files in os.walk(modelroot):
            for name in files:
                if name.lower().endswith(".off"):
                    path = os.path.join(root, name)
                    relpath = path[len(modelroot):]
                    self.filequeue.put(relpath)


    def hasNext(self) -> bool:
        return not self.filequeue.empty()

    def nextBatch(self):
        batch = torch.zeros(min(self.batchsize, self.filequeue.qsize()), self.pointcount, 3)
        for i in range(self.batchsize):
            if not self.filequeue.empty():
                filename = self.filequeue.get()
                objpath = os.path.join(self.modelroot, filename)
                pcpath = os.path.join(self.pcroot, filename)
                if os.path.exists(pcpath):
                    batch[i,:,:] = pointcloud.load_pc_from_off(pcpath)[0,:,:]
                else:
                    mesh = trimesh.load_mesh(objpath)
                    bb = mesh.bounding_box
                    samples, _ = trimesh.sample.sample_surface(mesh, self.pointcount)
                    samples *= 100.0 / np.max(bb.primitive.extents)
                    pointcloud.write_pc_to_off(pcpath, samples)
                    batch[i,:,:] = samples
        return batch