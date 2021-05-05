
cmake_minimum_required(VERSION 3.18)

project(gmeval LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ASan:
#string(APPEND CMAKE_CXX_FLAGS " -fno-omit-frame-pointer -fsanitize=address")
#string(APPEND CMAKE_LINKER_FLAGS" -fno-omit-frame-pointer -fsanitize=address")

#target_compile_options(kre -Wall -Wextra -Wpedantic -Werror)
string(APPEND CMAKE_CXX_FLAGS " -ffast-math -fno-finite-math-only")
string(APPEND CMAKE_CXX_FLAGS_RELEASE " -O4 -march=native")

# this will enable the lld (llvm) linker. it's parallel and much faster, but not installed by default.
# if it's not installed, you'll get errors, that openmp or other stuff is not installed (hard to track down)
#string(APPEND CMAKE_EXE_LINKER_FLAGS " -fuse-ld=lld")

set (MY_ANACONDA_PATH "C:/Users/SimonFraiss/anaconda3")

list(APPEND CMAKE_PREFIX_PATH "${MY_ANACONDA_PATH}/lib/site-packages/torch/")

find_package(OpenMP REQUIRED)
find_package(Torch REQUIRED)
add_compile_definitions(_HAS_STD_BYTE=0)
add_compile_definitions(NOMINMAX=1)

set(CMAKE_INCLUDE_CURRENT_DIR ON)


set(MY_INCLUDE_PATHS
    ${MY_ANACONDA_PATH}/include/
    ./external
    .
)
include_directories(SYSTEM ${MY_INCLUDE_PATHS} SYSTEM ${TORCH_INCLUDE_DIRS})

set(GMEVAL_HEADERS
    sampler.hpp
    eval.hpp
    gmslib/base.hpp
    gmslib/geom.hpp
    gmslib/parallel.hpp
    gmslib/pointindex.hpp
    gmslib/pointset.hpp
    gmslib/vec.hpp
)
set(GMEVAL_SOURCES
    
)

#add_executable(computeMixture ${GMEVAL_HEADERS} ${GMEVAL_SOURCES})
#target_link_libraries(computeMixture PUBLIC OpenMP::OpenMP_CXX)

set(PYTHON_BINDINGS_SOURCES
    pyeval.cpp
)
add_library(python_bindings ${PYTHON_BINDINGS_SOURCES} ${GMEVAL_HEADERS})
target_link_libraries(python_bindings PUBLIC torch OpenMP::OpenMP_CXX)
