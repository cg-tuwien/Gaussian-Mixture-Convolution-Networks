import platform

cuda_extra_cuda_cflags = ["-O3",  "--use_fast_math", "--std=c++14", "--expt-extended-lambda", "-DNDEBUG", "--default-stream per-thread"]
if platform.system() == "Windows":
    cuda_extra_cflags = ["/O2", "/fp:fast", "/std:c++14", "/DGPE_NO_CUDA_ERROR_CHECKING"]
    cpp_extra_cflags = ["/openmp", "/O2", "/fp:fast", "/std:c++14"]
else:
    cuda_extra_cflags = ["-O4", "-ffast-math", "-march=native", "--std=c++14"]
    cpp_extra_cflags = ["-fopenmp", "-O4", "-ffast-math", "-march=native", "--std=c++14", "-DGPE_NO_CUDA_ERROR_CHECKING"]
    cuda_extra_cuda_cflags.append("-Xcompiler -fopenmp")  # correct place?

