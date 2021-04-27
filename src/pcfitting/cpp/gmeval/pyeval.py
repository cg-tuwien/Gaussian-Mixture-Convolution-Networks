import os
import platform

import torch
import torch.utils.cpp_extension

source_dir = os.path.dirname(__file__)
# print(source_dir)

extra_include_paths = [source_dir, source_dir + "/..", source_dir + "/../ext", source_dir + "/../gmslib/src/gmslib"]
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


def eval_rmsd_unscaled(point_cloud_source: torch.Tensor, point_cloud_generated: torch.Tensor) -> (float, float, float, float, float):
    # Returns rmsd, md, stdev, max
    return bindings.eval_rmse_psnr(point_cloud_source, point_cloud_generated, False, False)


#returns rmsd, md
def calc_rmsd_to_itself(point_cloud: torch.Tensor) -> (float, float):
    return bindings.calc_rmsd_to_itself(point_cloud)

def cov_measure(point_cloud: torch.Tensor) -> (float, float):
    return bindings.cov_measure(point_cloud)
