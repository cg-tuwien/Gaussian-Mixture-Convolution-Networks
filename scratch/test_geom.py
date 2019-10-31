#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 17:28:54 2019

@author: madam
"""

import trimesh
import trimesh.sample

import pathlib
import subprocess
import concurrent.futures
import numpy as np
import re
import os

import config

file_ending_off_regexp = re.compile(r"\.off$", re.IGNORECASE)

n_points = config.n_points
shape_parent_path = config.shape_parent_path
pointcloud_parent_path = config.pointcloud_parent_path
gmm_parent_path = config.gmm_parent_path

pc2gmmCmd = "/home/madam/Documents/work/tuw/build-pc2gmm-Desktop_Qt_5_9_6_GCC_64bit-Release/pc2gmm"

def genData(shape_sub_path):
    shape_sub_path = shape_sub_path.strip()
    
    shapeMeshPath = str(shape_parent_path.joinpath(shape_sub_path))
    mesh = trimesh.load_mesh(shapeMeshPath)
    bb = mesh.bounding_box
#    print('loaded')
    
    samples, _ = trimesh.sample.sample_surface(mesh, n_points)
    samples *= 100.0 / np.max(bb.primitive.extents)
#    print('converted to pc')
    
    pc_path = pointcloud_parent_path.joinpath(shape_sub_path)
    pc_path.parent.mkdir(parents=True, exist_ok=True)
    if pc_path.exists():
        pc_path.unlink()     # deletes the file
    file  = pc_path.open(mode="w")
    file.write("OFF\n")
    file.write(f"{samples.shape[0]} 0 0\n") 
    for point in samples:
        file.write(f"{point[0]} {point[1]} {point[2]}\n")
    file.close()
    
    shape_gmm_path = gmm_parent_path.joinpath(file_ending_off_regexp.sub(".ply", shape_sub_path))
    shape_gmm_path.parent.mkdir(parents=True, exist_ok=True)
    if shape_gmm_path.exists():
        shape_gmm_path.unlink()    # deletes the file
    pc2gmmCall = f"{pc2gmmCmd} {shapeMeshPath} {shape_gmm_path} --nNNDistSamples=10 --useWeightedPotentials --alpha=3.0 --nLevels=10"
#    print(pc2gmmCall)
    os.system(pc2gmmCall)
    
    print(shape_sub_path)

shape_file_list = open(shape_parent_path.joinpath('fileList'))

with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
    executor.map(genData, shape_file_list)
#for f in shape_file_list:
#    genData(f)

print('done')


