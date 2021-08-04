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
string(APPEND CMAKE_CUDA_FLAGS " -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-extended-lambda --expt-relaxed-constexpr") #
string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler '-fPIC,-fopenmp,-ffast-math,-fno-finite-math-only'  --use_fast_math")
string(APPEND CMAKE_CUDA_FLAGS_RELEASE " --generate-line-info -Xcompiler '-O4,-march=native' -O3")
#string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -g -G")

# gcc 10 is buggy (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=100102)
string(APPEND CMAKE_CUDA_FLAGS " -ccbin /usr/bin/g++-9")

# ASan:
#string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler -fsanitize=address -Xcompiler -fno-omit-frame-pointer")
#string(APPEND CMAKE_CXX_FLAGS " -fno-omit-frame-pointer -fsanitize=address")
#string(APPEND CMAKE_LINKER_FLAGS" -fno-omit-frame-pointer -fsanitize=address")

#target_compile_options(kre -Wall -Wextra -Wpedantic -Werror)
string(APPEND CMAKE_CXX_FLAGS " -ffast-math -fno-finite-math-only")
string(APPEND CMAKE_CXX_FLAGS_RELEASE " -O4 -march=native")

# this will enable the lld (llvm) linker. it's parallel and much faster, but not installed by default.
# if it's not installed, you'll get errors, that openmp or other stuff is not installed (hard to track down)
#string(APPEND CMAKE_EXE_LINKER_FLAGS " -fuse-ld=lld")

set (MY_ANACONDA_PATH "/home/madam/bin/anaconda3")

list(APPEND CMAKE_PREFIX_PATH "${MY_ANACONDA_PATH}/")
list(APPEND CMAKE_PREFIX_PATH "${MY_ANACONDA_PATH}/lib/python3.8/site-packages/torch/")

find_package(OpenMP REQUIRED)
find_package(Torch REQUIRED)
find_package(Catch2 REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
add_compile_definitions(GMC_CMAKE_TEST_BUILD)
add_compile_definitions(QT_NO_KEYWORDS)
add_compile_definitions(GLM_FORCE_XYZW_ONLY)

add_compile_definitions(GPE_LIMIT_N_REDUCTION)
add_compile_definitions(GPE_ONLY_FLOAT)
add_compile_definitions(GPE_ONLY_2D)
find_package(QT NAMES Qt5 COMPONENTS Core Widgets REQUIRED)
find_package(Qt5 COMPONENTS Core Widgets Test REQUIRED)


set(MY_INCLUDE_PATHS
    ${MY_ANACONDA_PATH}/include/python3.8/
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
    lbvh/building.h
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
    util/gaussian.h
    util/gaussian_mixture.h
    util/glm.h
    util/grad/algorithms.h
    util/grad/common.h
    util/grad/glm.h
    util/grad/gaussian.h
    util/grad/mixture.h
    util/grad/scalar.h
    util/helper.h
    util/mixture.h
    util/output.h
    util/scalar.h)
add_library(util ${UTIL_HEADERS} util/dummy.cpp)

set(EVALUATE_INVERSED_HEADERS
    evaluate_inversed/implementations.h
    evaluate_inversed/evaluate_inversed.h
)
set(EVALUATE_INVERSED_SOURCES
    evaluate_inversed/parallel_implementation.cu
    evaluate_inversed/parallel_implementation_optimised_forward.cu
    evaluate_inversed/parallel_implementation_optimised_backward.cu
    evaluate_inversed/evaluate_inversed.cpp
    evaluate_inversed/cuda_bvh_implementation.cu
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


set(PIECES_HEADERS
    pieces/pieces.h
    pieces/integrate.h
    pieces/matrix_inverse.h
)
set(PIECES_SOURCES
    pieces/pieces.cpp
    pieces/integrate.cu
    pieces/matrix_inverse.cu
)
add_library(pieces ${PIECES_HEADERS} ${PIECES_SOURCES})
target_link_libraries(pieces PUBLIC OpenMP::OpenMP_CXX torch math lbvh common)

set(BVH_MHEM_FIT_HEADERS
    bvh_mhem_fit/Config.h
    bvh_mhem_fit/bindings.h
    bvh_mhem_fit/implementation.h
    bvh_mhem_fit/implementation_common.h
    bvh_mhem_fit/implementation_template_externs.h
    bvh_mhem_fit/implementation_backward.h
    bvh_mhem_fit/implementation_forward.h
)
set(BVH_MHEM_FIT_SOURCES
    bvh_mhem_fit/bindings.cpp
    bvh_mhem_fit/implementation_dispatch.cpp
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_2_float_2.cu
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_2_float_3.cu
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_2_double_2.cu
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_2_double_3.cu
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_4_float_2.cu
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_4_float_3.cu
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_4_double_2.cu
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_4_double_3.cu
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_8_float_2.cu
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_8_float_3.cu
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_8_double_2.cu
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_8_double_3.cu
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_16_float_2.cu
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_16_float_3.cu
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_16_double_2.cu
    bvh_mhem_fit/implementation_forward_instances/template_instance_implementation_forward_16_double_3.cu
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_2_float_2.cu
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_2_float_3.cu
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_2_double_2.cu
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_2_double_3.cu
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_4_float_2.cu
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_4_float_3.cu
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_4_double_2.cu
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_4_double_3.cu
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_8_float_2.cu
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_8_float_3.cu
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_8_double_2.cu
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_8_double_3.cu
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_16_float_2.cu
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_16_float_3.cu
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_16_double_2.cu
    bvh_mhem_fit/implementation_backward_instances/template_instance_implementation_backward_16_double_3.cu
)
add_library(bvh_mhem_fit ${BVH_MHEM_FIT_HEADERS} ${BVH_MHEM_FIT_SOURCES})
target_link_libraries(bvh_mhem_fit PUBLIC OpenMP::OpenMP_CXX torch math lbvh common pieces)

set(BVH_MHEM_FIT_ALPHA_HEADERS
    bvh_mhem_fit_alpha/Config.h
    bvh_mhem_fit_alpha/implementation.h
    bvh_mhem_fit_alpha/implementation_common.h
    bvh_mhem_fit_alpha/implementation_template_externs.h
    bvh_mhem_fit_alpha/implementation_autodiff_backward.h
    bvh_mhem_fit_alpha/implementation_backward.h
    bvh_mhem_fit_alpha/implementation_forward.h
)
set(BVH_MHEM_FIT_ALPHA_SOURCES
    bvh_mhem_fit_alpha/implementation_dispatch.cpp
    bvh_mhem_fit_alpha/implementation_autodiff_backward.cpp
    bvh_mhem_fit_alpha/implementation_forward_instances/template_instance_implementation_forward_2_float_2.cpp
    bvh_mhem_fit_alpha/implementation_forward_instances/template_instance_implementation_forward_2_float_3.cpp
    bvh_mhem_fit_alpha/implementation_forward_instances/template_instance_implementation_forward_2_double_2.cpp
    bvh_mhem_fit_alpha/implementation_forward_instances/template_instance_implementation_forward_2_double_3.cpp
    bvh_mhem_fit_alpha/implementation_forward_instances/template_instance_implementation_forward_4_float_2.cpp
    bvh_mhem_fit_alpha/implementation_forward_instances/template_instance_implementation_forward_4_float_3.cpp
    bvh_mhem_fit_alpha/implementation_forward_instances/template_instance_implementation_forward_4_double_2.cpp
    bvh_mhem_fit_alpha/implementation_forward_instances/template_instance_implementation_forward_4_double_3.cpp
    bvh_mhem_fit_alpha/implementation_forward_instances/template_instance_implementation_forward_8_float_2.cpp
    bvh_mhem_fit_alpha/implementation_forward_instances/template_instance_implementation_forward_8_float_3.cpp
    bvh_mhem_fit_alpha/implementation_forward_instances/template_instance_implementation_forward_8_double_2.cpp
    bvh_mhem_fit_alpha/implementation_forward_instances/template_instance_implementation_forward_8_double_3.cpp
    bvh_mhem_fit_alpha/implementation_forward_instances/template_instance_implementation_forward_16_float_2.cpp
    bvh_mhem_fit_alpha/implementation_forward_instances/template_instance_implementation_forward_16_float_3.cpp
    bvh_mhem_fit_alpha/implementation_forward_instances/template_instance_implementation_forward_16_double_2.cpp
    bvh_mhem_fit_alpha/implementation_forward_instances/template_instance_implementation_forward_16_double_3.cpp
    bvh_mhem_fit_alpha/implementation_backward_instances/template_instance_implementation_backward_2_float_2.cpp
    bvh_mhem_fit_alpha/implementation_backward_instances/template_instance_implementation_backward_2_float_3.cpp
    bvh_mhem_fit_alpha/implementation_backward_instances/template_instance_implementation_backward_2_double_2.cpp
    bvh_mhem_fit_alpha/implementation_backward_instances/template_instance_implementation_backward_2_double_3.cpp
    bvh_mhem_fit_alpha/implementation_backward_instances/template_instance_implementation_backward_4_float_2.cpp
    bvh_mhem_fit_alpha/implementation_backward_instances/template_instance_implementation_backward_4_float_3.cpp
    bvh_mhem_fit_alpha/implementation_backward_instances/template_instance_implementation_backward_4_double_2.cpp
    bvh_mhem_fit_alpha/implementation_backward_instances/template_instance_implementation_backward_4_double_3.cpp
    bvh_mhem_fit_alpha/implementation_backward_instances/template_instance_implementation_backward_8_float_2.cpp
    bvh_mhem_fit_alpha/implementation_backward_instances/template_instance_implementation_backward_8_float_3.cpp
    bvh_mhem_fit_alpha/implementation_backward_instances/template_instance_implementation_backward_8_double_2.cpp
    bvh_mhem_fit_alpha/implementation_backward_instances/template_instance_implementation_backward_8_double_3.cpp
    bvh_mhem_fit_alpha/implementation_backward_instances/template_instance_implementation_backward_16_float_2.cpp
    bvh_mhem_fit_alpha/implementation_backward_instances/template_instance_implementation_backward_16_float_3.cpp
    bvh_mhem_fit_alpha/implementation_backward_instances/template_instance_implementation_backward_16_double_2.cpp
    bvh_mhem_fit_alpha/implementation_backward_instances/template_instance_implementation_backward_16_double_3.cpp
)
add_library(bvh_mhem_fit_alpha ${BVH_MHEM_FIT_ALPHA_HEADERS} ${BVH_MHEM_FIT_ALPHA_SOURCES})
target_link_libraries(bvh_mhem_fit_alpha PUBLIC OpenMP::OpenMP_CXX torch math lbvh common)
target_compile_definitions(bvh_mhem_fit_alpha PUBLIC GPE_AUTODIFF)


set(CONVOLUTION_HEADERS
    convolution/bindings.h
    convolution/implementation.h
    convolution/implementation_template_externs.h
    convolution/implementation_backward.h
    convolution/implementation_forward.h
)
set(CONVOLUTION_SOURCES
    convolution/bindings.cpp
    convolution/implementation_dispatch.cpp
    convolution/implementation_forward_instances/template_instance_implementation_forward_1_float_2.cu
    convolution/implementation_forward_instances/template_instance_implementation_forward_1_float_3.cu
    convolution/implementation_forward_instances/template_instance_implementation_forward_1_double_2.cu
    convolution/implementation_forward_instances/template_instance_implementation_forward_1_double_3.cu
    convolution/implementation_backward_instances/template_instance_implementation_backward_1_float_2.cu
    convolution/implementation_backward_instances/template_instance_implementation_backward_1_float_3.cu
    convolution/implementation_backward_instances/template_instance_implementation_backward_1_double_2.cu
    convolution/implementation_backward_instances/template_instance_implementation_backward_1_double_3.cu
)
add_library(convolution ${CONVOLUTION_HEADERS} ${CONVOLUTION_SOURCES})
target_link_libraries(convolution PUBLIC OpenMP::OpenMP_CXX torch math common pieces)

set(CONVOLUTION_FITTING_HEADERS
    convolution_fitting/Config.h
    convolution_fitting/bindings.h
    convolution_fitting/implementation.h
    convolution_fitting/implementation_common.h
    convolution_fitting/implementation_template_externs.h
    convolution_fitting/implementation_backward.h
    convolution_fitting/implementation_forward.h
    convolution_fitting/Tree.h
)
set(CONVOLUTION_FITTING_SOURCES
    convolution_fitting/bindings.cpp
    convolution_fitting/implementation_dispatch.cpp
    convolution_fitting/Tree.cu
    convolution_fitting/implementation_forward_instances/template_instance_implementation_forward_1_float_2.cu
    convolution_fitting/implementation_forward_instances/template_instance_implementation_forward_1_float_3.cu
    convolution_fitting/implementation_forward_instances/template_instance_implementation_forward_1_double_2.cu
    convolution_fitting/implementation_forward_instances/template_instance_implementation_forward_1_double_3.cu
#    convolution_fitting/implementation_forward_instances/template_instance_implementation_forward_2_float_2.cu
#    convolution_fitting/implementation_forward_instances/template_instance_implementation_forward_2_float_3.cu
#    convolution_fitting/implementation_forward_instances/template_instance_implementation_forward_2_double_2.cu
#    convolution_fitting/implementation_forward_instances/template_instance_implementation_forward_2_double_3.cu
#    convolution_fitting/implementation_forward_instances/template_instance_implementation_forward_4_float_2.cu
#    convolution_fitting/implementation_forward_instances/template_instance_implementation_forward_4_float_3.cu
#    convolution_fitting/implementation_forward_instances/template_instance_implementation_forward_4_double_2.cu
#    convolution_fitting/implementation_forward_instances/template_instance_implementation_forward_4_double_3.cu
#    convolution_fitting/implementation_forward_instances/template_instance_implementation_forward_8_float_2.cu
#    convolution_fitting/implementation_forward_instances/template_instance_implementation_forward_8_float_3.cu
#    convolution_fitting/implementation_forward_instances/template_instance_implementation_forward_8_double_2.cu
#    convolution_fitting/implementation_forward_instances/template_instance_implementation_forward_8_double_3.cu
    convolution_fitting/implementation_backward_instances/template_instance_implementation_backward_1_float_2.cu
    convolution_fitting/implementation_backward_instances/template_instance_implementation_backward_1_float_3.cu
    convolution_fitting/implementation_backward_instances/template_instance_implementation_backward_1_double_2.cu
    convolution_fitting/implementation_backward_instances/template_instance_implementation_backward_1_double_3.cu
#    convolution_fitting/implementation_backward_instances/template_instance_implementation_backward_2_float_2.cu
#    convolution_fitting/implementation_backward_instances/template_instance_implementation_backward_2_float_3.cu
#    convolution_fitting/implementation_backward_instances/template_instance_implementation_backward_2_double_2.cu
#    convolution_fitting/implementation_backward_instances/template_instance_implementation_backward_2_double_3.cu
#    convolution_fitting/implementation_backward_instances/template_instance_implementation_backward_4_float_2.cu
#    convolution_fitting/implementation_backward_instances/template_instance_implementation_backward_4_float_3.cu
#    convolution_fitting/implementation_backward_instances/template_instance_implementation_backward_4_double_2.cu
#    convolution_fitting/implementation_backward_instances/template_instance_implementation_backward_4_double_3.cu
#    convolution_fitting/implementation_backward_instances/template_instance_implementation_backward_8_float_2.cu
#    convolution_fitting/implementation_backward_instances/template_instance_implementation_backward_8_float_3.cu
#    convolution_fitting/implementation_backward_instances/template_instance_implementation_backward_8_double_2.cu
#    convolution_fitting/implementation_backward_instances/template_instance_implementation_backward_8_double_3.cu
)
add_library(convolution_fitting ${CONVOLUTION_FITTING_HEADERS} ${CONVOLUTION_FITTING_SOURCES})
target_link_libraries(convolution_fitting PUBLIC OpenMP::OpenMP_CXX torch math lbvh common pieces)


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

add_executable(1_test_convolution_fitting convolution_fitting/test.cpp)
target_link_libraries(1_test_convolution_fitting PUBLIC OpenMP::OpenMP_CXX torch Qt5::Widgets convolution_fitting convolution evaluate_inversed math common)

set(MATRIX_INVERSE_HEADERS
)
set(MATRIX_INVERSE_SOURCES
    math/matrix_inverse_cuda.cpp
    math/matrix_inverse_cuda.cu
)
add_library(matrix_inverse ${MATRIX_INVERSE_HEADERS} ${MATRIX_INVERSE_SOURCES})
target_link_libraries(matrix_inverse PUBLIC OpenMP::OpenMP_CXX torch)


#unit tests:
set(UNITTESTS_HEADERS
    unittests/support.h
)
set(UNITTESTS_SOURCES
    unittests/algorithms.cpp
#    unittests/bvh_mhem_fit_alpha_vs_ad.cpp
#    unittests/bvh_mhem_fit_working_vs_alpha.cpp
#    unittests/bvh_mhem_fit_working_vs_binding.cpp
#    unittests/evaluate_inversed.cpp
    unittests/grad_algorithms.cpp
    unittests/grad_mixture.cpp
    unittests/main.cpp
    unittests/util_helpers.cpp
)
add_executable(0_unittests ${UNITTESTS_SOURCES} ${UNITTESTS_HEADERS})
target_link_libraries(0_unittests PUBLIC Catch2::Catch2 OpenMP::OpenMP_CXX torch Qt5::Widgets bvh_mhem_fit_alpha bvh_mhem_fit integrate evaluate_inversed math lbvh common)
target_compile_definitions(0_unittests PUBLIC GPE_AUTODIFF)

include(CTest)
include(Catch)
catch_discover_tests(0_unittests)

set(BENCHMARK_HEADERS
)
set(BENCHMARK_SOURCES
    benchmarks/main.cpp
    benchmarks/bvh_mhem_fit.cpp
    benchmarks/evaluate_inversed.cpp
)
add_executable(0_benchmarks ${BENCHMARK_SOURCES} ${BENCHMARK_HEADERS})
target_link_libraries(0_benchmarks PUBLIC Catch2::Catch2 OpenMP::OpenMP_CXX torch Qt5::Widgets bvh_mhem_fit integrate evaluate_inversed math lbvh common)

