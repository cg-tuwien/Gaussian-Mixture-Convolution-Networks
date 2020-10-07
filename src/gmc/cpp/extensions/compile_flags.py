import platform

cuda_extra_cuda_cflags = ["-O3",  "--use_fast_math", "--std=c++14", "--expt-extended-lambda", "-DNDEBUG"]
if platform.system() == "Windows":
    cuda_extra_cflags = ["/openmp", "/O2", "/fp:fast", "/std:c++17", "/DGPE_NO_CUDA_ERROR_CHECKING"]
    cpp_extra_cflags = ["/openmp", "/O2", "/fp:fast", "/std:c++17"]
    cuda_extra_cuda_cflags.append("-Xcompiler=/openmp,/O2,/fp:fast,/DGPE_NO_CUDA_ERROR_CHECKING")
else:
    cuda_extra_cflags = ["-O4", "-ffast-math", "-march=native", "--std=c++17"]
    cpp_extra_cflags = ["-fopenmp", "-O4", "-ffast-math", "-march=native", "--std=c++17", "-DGPE_NO_CUDA_ERROR_CHECKING"]
    cuda_extra_cuda_cflags.append("-Xcompiler -fopenmp")  # correct place?

