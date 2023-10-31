cmake_minimum_required(VERSION 3.18.0)

set(PROJECT_NAME neighbors_iris)

project(${PROJECT_NAME})

# Tune up various optimizations
# Enable CUDA support
option(ENABLE_CUDA "Enable CUDA optimizations" ON)
# Set up options for basic compile optimizations
option(ENABLE_AVX2 "Enable AVX2 optimizations" ON)
option(ENABLE_AVX512 "Enable AVX512 optimizations" OFF)
option(ENABLE_AMX "Enable AMX (Advanced Matrix Extensions) optimizations" OFF)
# Set up options for OpenMP support
option(ENABLE_OPENMP "Enable OpenMP" ON)

include(FetchContent)

FetchContent_Declare(
    cmake
    GIT_REPOSITORY https://github.com/mgorshkov/cmake.git
    GIT_TAG main
)
FetchContent_MakeAvailable(cmake)

# Apply global cmake options
include(${cmake_SOURCE_DIR}/Config.cmake)

FetchContent_Declare(
    sklearn
    GIT_REPOSITORY https://github.com/mgorshkov/sklearn.git
    GIT_TAG main
    CMAKE_ARGS
        -DENABLE_CUDA=${ENABLE_CUDA}
        -DENABLE_OPENMP=${ENABLE_OPENMP}
        -DENABLE_SSE2=${ENABLE_SSE2}
        -DENABLE_SSE3=${ENABLE_SSE3}
        -DENABLE_AVX=${ENABLE_AVX}
        -DENABLE_AVX2=${ENABLE_AVX2}
        -DENABLE_AVX512=${ENABLE_AVX512}
        -DENABLE_AMX=${ENABLE_AMX}
        -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
        -DDEBUG_INFO=${DEBUG_INFO}
)
FetchContent_MakeAvailable(sklearn)

include_directories(${sklearn_SOURCE_DIR}/include)

add_executable(${PROJECT_NAME})

target_sources(${PROJECT_NAME} PUBLIC main.cpp)

set (LIBS ${LIBS}
    pd
    sklearn
    pthread)

# Apply target-dependent cmake options
include(${cmake_SOURCE_DIR}/Target.cmake)

install(
    TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT ${PROJECT_NAME}
)
