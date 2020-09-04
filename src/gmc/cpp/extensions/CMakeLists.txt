cmake_minimum_required(VERSION 3.18)

project(gm_pytorch_extensions LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
set(CMAKE_CUDA_ARCHITECTURES 60)
# set(CMAKE_CUDA_FLAGS
#     "-D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options '-fPIC' -O3 --use_fast_math -std=c++14"
#     )
# string(APPEND CMAKE_CXX_FLAGS " -D_GLIBCXX_USE_CXX11_ABI=0")
string(APPEND CMAKE_CUDA_FLAGS " -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options '-fPIC' -O3 --use_fast_math")

# set(CMAKE_CXX_FLAGS -D_GLIBCXX_USE_CXX11_ABI=0 -O3 --use_fast_math)

add_compile_definitions(GMC_CMAKE_TEST_BUILD)

set (MY_ANACONDA_PATH "/home/madam/bin/anaconda3")

list(APPEND CMAKE_PREFIX_PATH "${MY_ANACONDA_PATH}/lib/python3.7/site-packages/torch/")


set(CUDNN_INCLUDE_PATH
   ${MY_ANACONDA_PATH}/pkgs/cudnn-7.6.4-cuda10.1_0/include/
   )
set (CUDNN_LIBRARY_PATH
   ${MY_ANACONDA_PATH}/pkgs/cudnn-7.6.4-cuda10.1_0/lib/libcudnn.so
   )

find_package(OpenMP REQUIRED)
find_package(Torch REQUIRED)

set(MY_INCLUDE_PATHS
    ${MY_ANACONDA_PATH}/include/python3.7m/
    ../glm
    .
)


include_directories(SYSTEM ${MY_INCLUDE_PATHS} SYSTEM ${TORCH_INCLUDE_DIRS})

set(HEADERS
    common.h
)

set(EVALUATE_SOURCES
    evaluate/evaluate_inversed_cpu.cpp
    evaluate/evaluate_inversed_cuda.cpp
    evaluate/evaluate_inversed_cuda.cu
    evaluate2/test_runner.cpp
)

add_executable(gm_evaluate ${HEADERS} ${EVALUATE_SOURCES})
target_link_libraries(gm_evaluate PUBLIC OpenMP::OpenMP_CXX torch)

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
target_link_libraries(gm_evaluate2 PUBLIC OpenMP::OpenMP_CXX torch)
