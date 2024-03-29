cmake_minimum_required(VERSION 3.13.0)

set(SKLEARN sklearn)

project(${SKLEARN} VERSION 0.0.3 DESCRIPTION "ML Methods from scikit-learn library")

set(CMAKE_CXX_STANDARD 20)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if (NOT DEFINED CMAKE_INSTALL_BINDIR)
    set(CMAKE_INSTALL_BINDIR "bin" CACHE PATH "user executables (bin)")
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/W4)
    add_definitions(-DNOMINMAX) # allow using min/max in the project
else()
    add_compile_options(-Wall -Wextra -pedantic)
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

option(ENABLE_SSE2 "Enable SSE2 optimizations" ON)
option(ENABLE_SSE3 "Enable SSE3 optimizations" ON)
option(ENABLE_SAVX "Enable AVX optimizations" ON)
option(ENABLE_AVX2 "Enable AVX2 optimizations" ON)
option(ENABLE_AVX512 "Enable AVX512 optimizations" OFF)

if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    if (ENABLE_AVX512)
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            message("GNU or Clang compiler, AVX512 optimization ON")
            add_compile_options(-O3 -ftree-vectorize -mavx512f -mavx512dq -mavx512bw -mavx512vbmi -mavx512vbmi2 -mavx512vl)
        elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
            message("Intel compiler, AVX512 optimization ON")
            add_compile_options(-Ofast -mavx512f -mavx512dq -mavx512bw -mavx512vbmi -mavx512vbmi2 -mavx512vl)
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            message("MSVC compiler, AVX512 optimization ON")
            add_compile_options(/O2 /arch:AVX512)
        endif()
    endif()
    if (ENABLE_AVX2)
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            message("GNU or Clang compiler, AVX2 optimization ON")
            add_compile_options(-O3 -ftree-vectorize -mavx2)
        elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
            message("Intel compiler, AVX2 optimization ON")
            add_compile_options(-Ofast -msse4.2 -axAVX2,CORE-AVX2)
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            message("MSVC compiler, AVX2 optimization ON")
            add_compile_options(/O2 /arch:AVX2)
        endif()
    endif()
    if (ENABLE_AVX)
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            message("GNU or Clang compiler, AVX optimization ON")
            add_compile_options(-O3 -ftree-vectorize -mavx)
        elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
            message("Intel compiler, AVX512 optimization ON")
            add_compile_options(-Ofast -axAVX -xAVX)
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            message("MSVC compiler, AVX512 optimization ON")
            add_compile_options(/O2 /arch:AVX)
        endif()
    endif()
    if (ENABLE_SSE3)
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            message("GNU or Clang compiler, SSE3 optimization ON")
            add_compile_options(-O3 -ftree-vectorize -msse3)
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            message("Intel compiler, SSE3 optimization ON")
            add_compile_options(-Ofast -axSSE3 -xSSE3)
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            message("MSVC compiler, SSE3 optimization ON")
            add_compile_options(/O2 /arch:SSE2) # VC doesn't support /arch:SSE3
        endif()
    endif()
    if (ENABLE_SSE2)
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            message("GNU or Clang compiler, SSE2 optimization ON")
            add_compile_options(-O3 -ftree-vectorize -msse2)
        elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
            message("Intel compiler, SSE2 optimization ON")
            add_compile_options(-Ofast -axSSE2 -xSSE2)
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            message("MSVC compiler, SSE2 optimization ON")
            if (NOT CMAKE_CL_64)
                add_compile_options(/O2 /arch:SSE2)
            endif()
        endif()
    endif()
endif()

include(FetchContent)

FetchContent_Declare(
    scipy
    GIT_REPOSITORY https://github.com/mgorshkov/scipy.git
    GIT_TAG main
)

FetchContent_MakeAvailable(scipy)

include_directories(include ${np_SOURCE_DIR}/include ${pd_SOURCE_DIR}/include ${scipy_SOURCE_DIR}/include)

add_subdirectory(src)

set(MAKE_DOC OFF)
set(MAKE_UNIT_TESTS ON)

if (MAKE_UNIT_TESTS)
    message(STATUS "Enable testing in ${SKLEARN}")
    add_subdirectory(unit_tests)
    include (CTest)
    enable_testing()
endif()

if (MAKE_DOC)
    message(STATUS "Enable doxygen in ${SKLEARN}")
    add_subdirectory(doc)
endif()

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/sklearn
    DESTINATION include
    COMPONENT ${SKLEARN_UNIT_TESTS}
)
