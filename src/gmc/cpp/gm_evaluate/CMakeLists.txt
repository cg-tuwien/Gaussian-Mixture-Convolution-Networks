cmake_minimum_required(VERSION 3.5)

project(gm_evaluate LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MY_INCLUDE_PATHS
    /home/madam/bin/anaconda3/lib/python3.7/site-packages/torch/include
    /home/madam/bin/anaconda3/lib/python3.7/site-packages/torch/include/torch/csrc/api/include/
    /home/madam/bin/anaconda3/include/python3.7m/
    ../glm
)

include_directories(SYSTEM ${MY_INCLUDE_PATHS})

set(HEADERS
    common.h
)

set(SOURCES
    gm_evaluate_inversed_cpu.cpp
    gm_evaluate_inversed_cuda.cpp
    gm_evaluate_inversed_cuda.cu
)

add_executable(gm_evaluate ${HEADERS} ${SOURCES})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(gm_evaluate PUBLIC OpenMP::OpenMP_CXX)
endif()
