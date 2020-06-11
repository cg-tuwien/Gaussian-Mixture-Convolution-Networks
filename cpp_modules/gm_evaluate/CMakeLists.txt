cmake_minimum_required(VERSION 3.5)

project(gm_evaluate LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(SYSTEM /home/madam/bin/anaconda3/lib/python3.7/site-packages/torch/include /home/madam/bin/anaconda3/lib/python3.7/site-packages/torch/include/torch/csrc/api/include/)

set(HEADERS
    common.h
)

set(SOURCES
    gm_evaluate_inversed_cpu.cpp
    gm_evaluate_inversed_cuda.cpp
    gm_evaluate_inversed_cuda.cu
)


add_executable(gm_evaluate ${HEADERS} ${SOURCES})
