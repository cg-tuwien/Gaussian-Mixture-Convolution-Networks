cmake_minimum_required(VERSION 3.18)

project(gm_pytorch_extensions LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
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
string(APPEND CMAKE_CUDA_FLAGS " -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-extended-lambda --expt-relaxed-constexpr --compiler-options '-fPIC'")
string(APPEND CMAKE_CUDA_FLAGS_RELEASE " --use_fast_math")
string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -g -G")

# set(CMAKE_CXX_FLAGS -D_GLIBCXX_USE_CXX11_ABI=0 -O3 --use_fast_math)

add_compile_definitions(GMC_CMAKE_TEST_BUILD)

set (MY_ANACONDA_PATH "/home/madam/bin/anaconda3")

list(APPEND CMAKE_PREFIX_PATH "${MY_ANACONDA_PATH}/lib/python3.7/site-packages/torch/")





set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(OpenMP REQUIRED)
find_package(Torch REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
ADD_DEFINITIONS(-DQT_NO_KEYWORDS)
find_package(QT NAMES Qt5 COMPONENTS Core Widgets REQUIRED)
find_package(Qt5 COMPONENTS Core Widgets REQUIRED)

# https://gitlab.kitware.com/cmake/cmake/-/issues/16915
if ( TARGET Qt5::Core )
    get_property( core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS )
    string( REPLACE "-fPIC" "" new_core_options "${core_options}" )
    set_property( TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options} )
    set_property( TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON" )
    set( CMAKE_CXX_COMPILE_OPTIONS_PIE "-fPIC" )
endif()


set(MY_INCLUDE_PATHS
    ${MY_ANACONDA_PATH}/include/python3.7m/
    ../glm
    .
)


include_directories(SYSTEM ${MY_INCLUDE_PATHS} SYSTEM ${TORCH_INCLUDE_DIRS})

set(HEADERS
    common.h
)

set(LBVH_HEADERS
    lbvh/aabb.h
    lbvh/bvh.h
    lbvh/morton_code.h
    lbvh/predicator.h
    lbvh/query.h
    lbvh/utility.h)



set(MATH_SOURCES
    math/symeig.h
    math/symeig_detail.h
    math/scalar.h
    math/matrix.h
    math/symeig_cpu.cpp
    math/symeig_cuda.cpp
    math/symeig_cuda.cu)

set(EVALUATE_SOURCES
    evaluate_inversed/cpu_bvh.cpp
    evaluate_inversed/cuda_bvh.cpp
    evaluate_inversed/cuda_bvh.cu
    evaluate_inversed/cpu_parallel.cpp
    evaluate_inversed/cuda_parallel.cpp
    evaluate_inversed/cuda_parallel.cu
    evaluate_inversed/test_runner.cpp
)

add_executable(evaluate_inversed_tester ${HEADERS} ${MATH_SOURCES} ${LBVH_HEADERS} ${EVALUATE_SOURCES})
target_link_libraries(evaluate_inversed_tester PUBLIC OpenMP::OpenMP_CXX torch Qt5::Widgets)

#set(EM_FITTING_SOURCES
#    em_fitting/em_fitting_common.h
#    em_fitting/em_fitting_cpu.cpp
#    em_fitting/em_fitting_cuda.cpp
#    em_fitting/em_fitting_cuda.cu
#)

#add_executable(em_fitting ${HEADERS} ${EM_FITTING_SOURCES})
#target_link_libraries(em_fitting PUBLIC OpenMP::OpenMP_CXX ${PYTORCH_LIBS})

###### add to python pytorch cuda flags:
# --expt-extended-lambda
