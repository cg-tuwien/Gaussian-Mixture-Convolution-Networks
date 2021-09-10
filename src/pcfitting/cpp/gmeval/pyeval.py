import os
import platform

import torch
import torch.utils.cpp_extension

source_dir = os.path.dirname(__file__)
# print(source_dir)

extra_include_paths = [source_dir, source_dir + "/..", source_dir + "/external", source_dir + "/../external/eigen", source_dir + "/../gmslib/src/gmslib"]
if platform.system() == "Windows":
    cpp_extra_cflags = ["/openmp", "/O2", "/std:c++17", "/DNDEBUG", "/D_HAS_STD_BYTE=0", "/DNOMINMAX"]
else:
    cpp_extra_cflags = ["-fopenmp", "-ffast-math", " -fno-finite-math-only", "-O4", "-march=native", "--std=c++17", "-DNDEBUG"]

bindings = torch.utils.cpp_extension.load('gmeval',
                [source_dir + '/pyeval.cpp'],
                extra_include_paths=extra_include_paths, verbose=True, extra_cflags=cpp_extra_cflags, extra_ldflags=["-lpthread"])


def eval_psnr(point_cloud_source: torch.Tensor, point_cloud_generated: torch.Tensor) -> float:
    return bindings.eval_rmse_psnr(point_cloud_source, point_cloud_generated, True, True)[0]


def eval_rmse(point_cloud_source: torch.Tensor, point_cloud_generated: torch.Tensor) -> float:
    return bindings.eval_rmse_psnr(point_cloud_source, point_cloud_generated, True, False)[0]


def eval_rmsd_unscaled(point_cloud_source: torch.Tensor, point_cloud_generated: torch.Tensor) -> (float, float, float, float):
    # Returns rmsd, md, stdev, max
    print("Note: consider calling eval_rmsd_both_sides!")
    return bindings.eval_rmse_psnr(point_cloud_source, point_cloud_generated, False, False)


#returns rmsd, md
def calc_rmsd_to_itself(point_cloud: torch.Tensor) -> (float, float):
    return bindings.calc_rmsd_to_itself(point_cloud)

def cov_measure(point_cloud: torch.Tensor) -> (float, float):
    return bindings.cov_measure(point_cloud)

def sample_gmm(gmm: torch.Tensor, count: int) -> torch.Tensor:
    return bindings.sample_gmm(gmm, count)

def eval_rmsd_both_sides(point_cloud_source: torch.Tensor, point_cloud_generated: torch.Tensor) -> (float, float, float, float, float, float, float, float):
    return bindings.eval_rmsd_both_sides(point_cloud_source, point_cloud_generated)

def calc_std_1_5(point_cloud_source: torch.Tensor, point_cloud_generated: torch.Tensor) -> (float, float, float, float):
    return bindings.calc_std_1_5(point_cloud_source, point_cloud_generated)

def cov_measure_5(point_cloud: torch.Tensor) -> (float, float):
    return bindings.cov_measure_5(point_cloud)

def avg_kl_div(gmm: torch.Tensor) -> float:
    return bindings.avg_kl_div(gmm)

def nn_graph(pointcloud: torch.Tensor, ncount: int) -> torch.Tensor:
    return bindings.nn_graph(pointcloud, ncount)

def nn_graph_sub(pointcloud: torch.Tensor, samplecount: int, ncount: int) -> torch.Tensor:
    return bindings.nn_graph_sub(pointcloud, samplecount, ncount)

def smoothness(responsibilities: torch.Tensor, nngraph: torch.Tensor) -> float:
    return bindings.smoothness(responsibilities, nngraph)

def irregularity(densities: torch.Tensor, nngraph: torch.Tensor) -> float:
    return bindings.irregularity(densities, nngraph)

def irregularity_sub(densities: torch.Tensor, nngraph: torch.Tensor) -> float:
    return bindings.irregularity_sub(densities, nngraph)