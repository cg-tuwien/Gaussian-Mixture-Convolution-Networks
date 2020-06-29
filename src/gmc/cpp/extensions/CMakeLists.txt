cmake_minimum_required(VERSION 3.5)

project(gm_pytorch_extensions LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

set(MY_INCLUDE_PATHS
    /home/madam/bin/anaconda3/lib/python3.7/site-packages/torch/include
    /home/madam/bin/anaconda3/lib/python3.7/site-packages/torch/include/torch/csrc/api/include/
    /home/madam/bin/anaconda3/include/python3.7m/
    ../glm
    .
)

include_directories(SYSTEM ${MY_INCLUDE_PATHS})

set(HEADERS
    common.h
)

set(EVALUATE_SOURCES
    evaluate/evaluate_inversed_cpu.cpp
    evaluate/evaluate_inversed_cuda.cpp
    evaluate/evaluate_inversed_cuda.cu
)

add_executable(gm_evaluate ${HEADERS} ${EVALUATE_SOURCES})
target_link_libraries(gm_evaluate PUBLIC OpenMP::OpenMP_CXX)

set(EM_FITTING_SOURCES
    em_fitting/em_fitting_cpu.cpp
    em_fitting/em_fitting_cuda.cpp
    em_fitting/em_fitting_cuda.cu
)

add_executable(em_fitting ${HEADERS} ${EM_FITTING_SOURCES})
target_link_libraries(em_fitting PUBLIC OpenMP::OpenMP_CXX)

