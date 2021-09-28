import platform
import os

source_dir = os.path.dirname(__file__)
print(f"compile_flags.py: source_dir={source_dir}")
extra_include_paths = [source_dir + "/../glm/", source_dir + "/", source_dir + "/../yamc/include", source_dir + "/../gcem/include", source_dir + "/.."]  # source_dir + "/../cub/",

cuda_extra_cuda_cflags = ["-O3",  "--use_fast_math", "--expt-extended-lambda", "-std=c++17", " --expt-relaxed-constexpr", "-DGPE_LIMIT_N_REDUCTION", "-DNDEBUG"]  # , "-DNDEBUG"
if platform.system() == "Windows":
    cuda_extra_cflags = ["/openmp", "/O2", "/fp:fast", "/DGPE_NO_CUDA_ERROR_CHECKING", "/DNDEBUG", "/DGPE_LIMIT_N_REDUCTION"]
    cpp_extra_cflags = ["/openmp", "/O2", "/fp:fast", "/std:c++17", "/DNDEBUG", "/DGPE_LIMIT_N_REDUCTION"]
    cuda_extra_cuda_cflags.append("-Xcompiler=/openmp,/O2,/fp:fast,/DGPE_NO_CUDA_ERROR_CHECKING")
else:
    cuda_extra_cflags = ["-O3", "-ffast-math", "-march=native", "-std=c++17", "-DGPE_LIMIT_N_REDUCTION", "-DNDEBUG"]
    cpp_extra_cflags = ["-fopenmp", "-ffast-math", " -fno-finite-math-only", "-O4", "-march=native", "--std=c++17", "-DGPE_LIMIT_N_REDUCTION", "-DNDEBUG"]  # , "-DNDEBUG", "-DGPE_NO_CUDA_ERROR_CHECKING"
    cuda_extra_cuda_cflags.append("-Xcompiler -fopenmp -ccbin /usr/bin/g++-9 -DNDEBUG")  #  -DNDEBUG"

