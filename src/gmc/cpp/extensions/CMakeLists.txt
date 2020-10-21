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
string(APPEND CMAKE_CUDA_FLAGS " -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-extended-lambda --expt-relaxed-constexpr")
string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler -fPIC,-fopenmp")
string(APPEND CMAKE_CUDA_FLAGS_RELEASE " --use_fast_math --generate-line-info")
string(APPEND CMAKE_CUDA_FLAGS_RELEASE " -Xcompiler '-ffast-math,-O4,-march=native'")
string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -g -G")

# ASan:
#string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -Xcompiler -fsanitize=address -Xcompiler -fno-omit-frame-pointer")
#string(APPEND CMAKE_CXX_FLAGS_DEBUG " -fno-omit-frame-pointer -fsanitize=address")
#string(APPEND CMAKE_LINKER_FLAGS_DEBUG " -fno-omit-frame-pointer -fsanitize=address")

#target_compile_options(kre -Wall -Wextra -Wpedantic -Werror)
string(APPEND CMAKE_CXX_FLAGS_RELEASE " -ffast-math -O4 -march=native")

# this will enable the lld (llvm) linker. it's parallel and much faster, but not installed by default.
# if it's not installed, you'll get errors, that openmp or other stuff is not installed (hard to track down)
#string(APPEND CMAKE_EXE_LINKER_FLAGS " -fuse-ld=lld")

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
    ../cub
    ../yamc/include
    .
)


include_directories(SYSTEM ${MY_INCLUDE_PATHS} SYSTEM ${TORCH_INCLUDE_DIRS})

set(COMMON_SOURCES
    common.h
    mixture.h
    parallel_start.h
    hacked_accessor.h
    cuda_qt_creator_definitinos.h
    cuda_operations.h
    CpuSynchronisationPoint.h
    CpuSynchronisationPoint.cpp
    ParallelStack.h
)

set(LBVH_SOURCES
    lbvh/aabb.h
    lbvh/bvh.h
    lbvh/bvh.cu
    lbvh/morton_code.h
    lbvh/predicator.h
    lbvh/query.h
    lbvh/utility.h)

set(MATH_SOURCES
    math/symeig.h
    math/symeig_cuda.h
    math/symeig_cpu.h
    math/symeig_detail.h
    math/scalar.h
    math/matrix.h
    math/symeig_cpu.cpp
    math/symeig_cuda.cpp
    math/symeig_cuda.cu)

set(EVALUATE_SOURCES
    evaluate_inversed/parallel_implementation.h
    evaluate_inversed/parallel_implementation.cu
    evaluate_inversed/parallel_implementation_optimised_forward.cu
    evaluate_inversed/parallel_implementation_optimised_backward.cu
    evaluate_inversed/parallel_binding.h
    evaluate_inversed/parallel_binding.cpp
    evaluate_inversed/cuda_bvh.cpp
    evaluate_inversed/cuda_bvh.cu
)

set(BVH_MHEM_FIT_SOURCES
    bvh_mhem_fit/implementation.h
    bvh_mhem_fit/implementation_forward.cu
    bvh_mhem_fit/implementation_backward.cu
    bvh_mhem_fit/implementation_common.cuh
    bvh_mhem_fit/bindings.cpp
    bvh_mhem_fit/bindings.h
)

add_executable(evaluate_inversed_tester ${COMMON_SOURCES} ${MATH_SOURCES} ${LBVH_SOURCES} ${EVALUATE_SOURCES} evaluate_inversed/test_runner.cpp)
target_link_libraries(evaluate_inversed_tester PUBLIC OpenMP::OpenMP_CXX torch Qt5::Widgets)

add_executable(bvh_mhem_fit_tester ${COMMON_SOURCES} ${MATH_SOURCES} ${LBVH_SOURCES} ${BVH_MHEM_FIT_SOURCES} ${EVALUATE_SOURCES} bvh_mhem_fit/bvh_fit_tester.cpp)
target_link_libraries(bvh_mhem_fit_tester PUBLIC OpenMP::OpenMP_CXX torch Qt5::Widgets)

#set(EM_FITTING_SOURCES
#    em_fitting/em_fitting_common.h
#    em_fitting/em_fitting_cpu.cpp
#    em_fitting/em_fitting_cuda.cpp
#    em_fitting/em_fitting_cuda.cu
#)

#add_executable(em_fitting ${COMMON_SOURCES} ${EM_FITTING_SOURCES})
#target_link_libraries(em_fitting PUBLIC OpenMP::OpenMP_CXX ${PYTORCH_LIBS})

