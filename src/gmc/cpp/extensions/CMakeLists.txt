cmake_minimum_required(VERSION 3.18)

project(gm_pytorch_extensions LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
set(CMAKE_CUDA_ARCHITECTURES 60)
set(CMAKE_CUDA_FLAGS
    "-D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options '-fPIC' -O3 --use_fast_math -std=c++14"
    )
string(APPEND CMAKE_CXX_FLAGS " -D_GLIBCXX_USE_CXX11_ABI=0")

#set(CMAKE_CXX_FLAGS -D_GLIBCXX_USE_CXX11_ABI=0 -O3 --use_fast_math)

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

add_compile_definitions(GMC_CMAKE_TEST_BUILD)

# only works in conda environment
#set(Torch_DIR
#    $ENV{CONDA_PREFIX}/lib/python3.7/site-packages/torch/share/cmake/Torch/
#)

set(Torch_DIR
    /home/madam/bin/anaconda3/lib/python3.7/site-packages/torch/share/cmake/Torch/
)
set(CUDNN_INCLUDE_PATH
    /home/madam/bin/anaconda3/pkgs/cudnn-7.6.4-cuda10.1_0/include/
    )
set (CUDNN_LIBRARY_PATH
    /home/madam/bin/anaconda3/pkgs/cudnn-7.6.4-cuda10.1_0/lib/
    )

find_package(OpenMP REQUIRED)
#find_package(Torch REQUIRED)

set(MY_INCLUDE_PATHS
    /home/madam/bin/anaconda3/lib/python3.7/site-packages/torch/include
    /home/madam/bin/anaconda3/lib/python3.7/site-packages/torch/include/torch/csrc/api/include/
    /home/madam/bin/anaconda3/include/python3.7m/
    ../glm
    .
)

 set(PYTORCH_LIBS
     c10_cuda
     c10
#     caffe2_detectron_ops_gpu
#     caffe2_module_test_dynamic
#     caffe2_nvrtc
#     caffe2_observers
#     shm
     torch_cpu
     torch_cuda
#     torch_global_deps
     torch_python
     torch
     cudart
 )

include_directories(SYSTEM ${MY_INCLUDE_PATHS})
link_directories(/home/madam/bin/anaconda3/lib/python3.7/site-packages/torch/lib)

set(HEADERS
    common.h
)

#set(EVALUATE_SOURCES
#    evaluate/evaluate_inversed_cpu.cpp
#    evaluate/evaluate_inversed_cuda.cpp
#    evaluate/evaluate_inversed_cuda.cu
#)

#add_executable(gm_evaluate ${HEADERS} ${EVALUATE_SOURCES})
#target_link_libraries(gm_evaluate PUBLIC OpenMP::OpenMP_CXX ${PYTORCH_LIBS})

#set(EM_FITTING_SOURCES
#    em_fitting/em_fitting_common.h
#    em_fitting/em_fitting_cpu.cpp
#    em_fitting/em_fitting_cuda.cpp
#    em_fitting/em_fitting_cuda.cu
#)

#add_executable(em_fitting ${HEADERS} ${EM_FITTING_SOURCES})
#target_link_libraries(em_fitting PUBLIC OpenMP::OpenMP_CXX ${PYTORCH_LIBS})


set(EVALUATE2_SOURCES
    evaluate2/evaluate2_inversed_cpu.cpp
    evaluate2/evaluate2_inversed_cuda.cpp
    evaluate2/evaluate2_inversed_cuda.cu
    evaluate2/test_runner.cpp
)

add_executable(gm_evaluate2 ${HEADERS} ${EVALUATE2_SOURCES})
target_link_libraries(gm_evaluate2 PUBLIC OpenMP::OpenMP_CXX ${PYTORCH_LIBS})
