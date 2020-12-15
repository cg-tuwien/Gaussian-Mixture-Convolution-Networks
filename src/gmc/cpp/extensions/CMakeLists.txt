cmake_minimum_required(VERSION 3.18)

project(gm_pytorch_extensions LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
set(CMAKE_CUDA_ARCHITECTURES 75)
# set(CMAKE_CUDA_FLAGS
#     "-D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options '-fPIC' -O3 --use_fast_math -std=c++14"
#     )
string(APPEND CMAKE_CUDA_FLAGS " -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-extended-lambda --expt-relaxed-constexpr")
string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler '-fPIC,-fopenmp,-ffast-math'  --use_fast_math")
string(APPEND CMAKE_CUDA_FLAGS_RELEASE " --generate-line-info -Xcompiler '-O4,-march=native'")
string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -g -G")

# ASan:
#string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler -fsanitize=address -Xcompiler -fno-omit-frame-pointer")
#string(APPEND CMAKE_CXX_FLAGS " -fno-omit-frame-pointer -fsanitize=address")
#string(APPEND CMAKE_LINKER_FLAGS" -fno-omit-frame-pointer -fsanitize=address")

#target_compile_options(kre -Wall -Wextra -Wpedantic -Werror)
string(APPEND CMAKE_CXX_FLAGS " -ffast-math")
string(APPEND CMAKE_CXX_FLAGS_RELEASE " -O4 -march=native")

# this will enable the lld (llvm) linker. it's parallel and much faster, but not installed by default.
# if it's not installed, you'll get errors, that openmp or other stuff is not installed (hard to track down)
#string(APPEND CMAKE_EXE_LINKER_FLAGS " -fuse-ld=lld")

set (MY_ANACONDA_PATH "/home/madam/bin/anaconda3")

list(APPEND CMAKE_PREFIX_PATH "${MY_ANACONDA_PATH}/lib/python3.7/site-packages/torch/")

find_package(OpenMP REQUIRED)
find_package(Torch REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
add_compile_definitions(GMC_CMAKE_TEST_BUILD)
add_compile_definitions(QT_NO_KEYWORDS)
add_compile_definitions(GLM_FORCE_XYZW_ONLY)
add_compile_definitions(GPE_LIMIT_N_REDUCTION)
add_compile_definitions(GPE_ONLY_FLOAT)
add_compile_definitions(GPE_ONLY_2D)
find_package(QT NAMES Qt5 COMPONENTS Core Widgets REQUIRED)
find_package(Qt5 COMPONENTS Core Widgets REQUIRED)

set(MY_INCLUDE_PATHS
    ${MY_ANACONDA_PATH}/include/python3.7m/
    ../glm
    ../yamc/include
    ../gcem/include
    ../autodiff
    .
)
include_directories(SYSTEM ${MY_INCLUDE_PATHS} SYSTEM ${TORCH_INCLUDE_DIRS})

set(COMMON_HEADERS
    common.h
    parallel_start.h
    hacked_accessor.h
    cuda_qt_creator_definitinos.h
    cuda_operations.h
    CpuSynchronisationPoint.h
    ParallelStack.h
)
set(COMMON_SOURCES
    CpuSynchronisationPoint.cpp
    unit_tests.cpp
)
add_library(common ${COMMON_HEADERS} ${COMMON_SOURCES})

set(MATH_HEADERS
    math/symeig.h
    math/symeig_cuda.h
    math/symeig_cpu.h
    math/symeig_detail.h)
set(MATH_SOURCES
    math/symeig_cpu.cpp
    math/symeig_cuda.cpp
    math/symeig_cuda.cu)
add_library(math ${MATH_HEADERS} ${MATH_SOURCES})
target_link_libraries(math PUBLIC OpenMP::OpenMP_CXX torch common)

set(LBVH_HEADERS
    lbvh/aabb.h
    lbvh/bvh.h
    lbvh/Config.h
    lbvh/morton_code.h
    lbvh/predicator.h
    lbvh/query.h
    lbvh/utility.h)
add_library(lbvh ${LBVH_HEADERS} lbvh/bvh.cu)
target_link_libraries(lbvh PUBLIC OpenMP::OpenMP_CXX torch math common)

set(UTIL_HEADERS
    util/algorithms.h
    util/autodiff.h
    util/containers.h
    util/cuda.h
    util/epsilon.h
    util/glm.h
    util/mixture.h
    util/grad/algorithms.h
    util/grad/common.h
    util/grad/glm.h
    util/grad/mixture.h
    util/scalar.h)
add_library(util ${UTIL_HEADERS} util/dummy.cpp)

set(EVALUATE_INVERSED_HEADERS
    evaluate_inversed/parallel_implementation.h
    evaluate_inversed/parallel_binding.h
)
set(EVALUATE_INVERSED_SOURCES
    evaluate_inversed/parallel_implementation.cu
    evaluate_inversed/parallel_implementation_optimised_forward.cu
    evaluate_inversed/parallel_implementation_optimised_backward.cu
    evaluate_inversed/parallel_binding.cpp
    evaluate_inversed/cuda_bvh.cpp
    evaluate_inversed/cuda_bvh.cu
)
add_library(evaluate_inversed ${EVALUATE_INVERSED_HEADERS} ${EVALUATE_INVERSED_SOURCES})
target_link_libraries(evaluate_inversed PUBLIC OpenMP::OpenMP_CXX torch math lbvh common)

set(INTEGRATE_HEADERS
    integrate/implementation.h
    integrate/binding.h
)
set(INTEGRATE_SOURCES
    integrate/implementation.cu
    integrate/binding.cpp
)
add_library(integrate ${INTEGRATE_HEADERS} ${INTEGRATE_SOURCES})
target_link_libraries(integrate PUBLIC OpenMP::OpenMP_CXX torch math lbvh common)

set(BVH_MHEM_FIT_HEADERS
    bvh_mhem_fit/BvhMhemFitConfig.h
    bvh_mhem_fit/bindings.h
    bvh_mhem_fit/implementation.h
    bvh_mhem_fit/implementation_common.h
    bvh_mhem_fit/implementation_template_externs.h
    bvh_mhem_fit/implementation_autodiff_backward.h
    bvh_mhem_fit/implementation_backward.h
    bvh_mhem_fit/implementation_forward.h
)
set(BVH_MHEM_FIT_SOURCES
    bvh_mhem_fit/bindings.cpp
    bvh_mhem_fit/implementation_dispatch.cpp
    bvh_mhem_fit/implementation_autodiff_backward.cpp
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_2_float_2.cpp
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_2_float_3.cpp
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_2_double_2.cpp
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_2_double_3.cpp
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_4_float_2.cpp
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_4_float_3.cpp
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_4_double_2.cpp
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_4_double_3.cpp
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_8_float_2.cpp
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_8_float_3.cpp
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_8_double_2.cpp
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_8_double_3.cpp
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_16_float_2.cpp
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_16_float_3.cpp
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_16_double_2.cpp
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_16_double_3.cpp
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_2_float_2.cpp
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_2_float_3.cpp
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_2_double_2.cpp
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_2_double_3.cpp
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_4_float_2.cpp
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_4_float_3.cpp
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_4_double_2.cpp
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_4_double_3.cpp
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_8_float_2.cpp
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_8_float_3.cpp
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_8_double_2.cpp
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_8_double_3.cpp
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_16_float_2.cpp
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_16_float_3.cpp
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_16_double_2.cpp
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_16_double_3.cpp
)
add_library(bvh_mhem_fit ${BVH_MHEM_FIT_HEADERS} ${BVH_MHEM_FIT_SOURCES})
target_link_libraries(bvh_mhem_fit PUBLIC OpenMP::OpenMP_CXX torch math lbvh common)


# https://gitlab.kitware.com/cmake/cmake/-/issues/16915
if ( TARGET Qt5::Core )
    get_property( core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS )
    string( REPLACE "-fPIC" "" new_core_options "${core_options}" )
    set_property( TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options} )
    set_property( TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON" )
    set( CMAKE_CXX_COMPILE_OPTIONS_PIE "-fPIC" )
endif()

add_executable(0_test_evaluate_inversed evaluate_inversed/test_evaluate_inversed.cpp)
target_link_libraries(0_test_evaluate_inversed PUBLIC OpenMP::OpenMP_CXX torch Qt5::Widgets evaluate_inversed math lbvh common)

add_executable(0_test_bvh_fit bvh_mhem_fit/test_bvh_fit.cpp)
target_link_libraries(0_test_bvh_fit PUBLIC OpenMP::OpenMP_CXX torch Qt5::Widgets bvh_mhem_fit integrate evaluate_inversed math lbvh common)

add_executable(0_unittest_bvh_fit_backward bvh_mhem_fit/unittest_mixture_grad.cpp bvh_mhem_fit/unittest_algorithms_grad.cpp bvh_mhem_fit/unittest_bvh_fit_backward.cpp)
target_link_libraries(0_unittest_bvh_fit_backward PUBLIC OpenMP::OpenMP_CXX torch Qt5::Widgets bvh_mhem_fit integrate evaluate_inversed math lbvh common)

