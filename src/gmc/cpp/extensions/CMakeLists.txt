cmake_minimum_required(VERSION 3.16)

project(gm_pytorch_extensions LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_70,code=sm_70 -O3 --use_fast_math")
string(APPEND CMAKE_CXX_FLAGS " -D_GLIBCXX_USE_CXX11_ABI=1")


add_compile_definitions(GMC_CMAKE_TEST_BUILD)

find_package(OpenMP REQUIRED)

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
    caffe2_detectron_ops_gpu
    caffe2_module_test_dynamic
    caffe2_nvrtc
    caffe2_observers
    shm
    torch_cpu
    torch_cuda
    torch_global_deps
    torch_python
    torch
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
