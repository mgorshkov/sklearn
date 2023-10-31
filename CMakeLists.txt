cmake_minimum_required(VERSION 3.18.0)

# Project name
set(PROJECT_NAME sklearn)

# Set up project
project(${PROJECT_NAME} VERSION 0.0.4 DESCRIPTION "⚡ ML methods in C++ | CUDA GPU + SIMD (AVX2/AVX512/AMX) CPU")

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
    scipy
    GIT_REPOSITORY https://github.com/mgorshkov/scipy.git
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

FetchContent_MakeAvailable(scipy)

include_directories(include ${np_SOURCE_DIR}/include ${pd_SOURCE_DIR}/include ${scipy_SOURCE_DIR}/include)

add_subdirectory(src)

# Apply target-dependent cmake options
include(${cmake_SOURCE_DIR}/Target.cmake)

set(MAKE_DOC OFF)
set(MAKE_UNIT_TESTS ON)

if (MAKE_UNIT_TESTS)
    message(STATUS "Enable testing in ${PROJECT_NAME}")
    add_subdirectory(unit_tests)
    include (CTest)
    enable_testing()
endif()

if (MAKE_DOC)
    message(STATUS "Enable doxygen in ${PROJECT_NAME}")
    add_subdirectory(doc)
endif()

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/sklearn
    DESTINATION include
    COMPONENT ${PROJECT_NAME}
)
