cmake_minimum_required(VERSION 3.13.0)

set(GMT_TREND_2D gmt_trend_2d)

project(${GMT_TREND_2D})

set(CMAKE_CXX_STANDARD 20)

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/W4)
    add_compile_definitions(NOMINMAX) # allow using min/max in the project
else()
    add_compile_options(-Wall -Wextra -pedantic)
endif()

option(ENABLE_OPENMP "Enable OpenMP" OFF)
if (ENABLE_OPENMP)
    message("OpenMP ON")
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

option(ENABLE_SSE2 "Enable SSE2 optimizations" OFF)
option(ENABLE_SSE3 "Enable SSE3 optimizations" OFF)
option(ENABLE_AVX "Enable AVX optimizations" OFF)
option(ENABLE_AVX2 "Enable AVX2 optimizations" OFF)
option(ENABLE_AVX512 "Enable AVX512 optimizations" OFF)

if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    if (ENABLE_AVX512)
        add_compile_definitions(ENABLE_AVX512)
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            message("GNU or Clang compiler, AVX512 optimization ON")
            add_compile_options(-O3 -ftree-vectorize -mavx512f -mavx512dq -mavx512bw -mavx512vbmi -mavx512vbmi2 -mavx512vl)
        elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
            message("Intel compiler, AVX512 optimization ON")
            add_compile_options(-O3 -mavx512f -mavx512dq -mavx512bw -mavx512vbmi -mavx512vbmi2 -mavx512vl)
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            message("MSVC compiler, AVX512 optimization ON")
            add_compile_options(/O2 /arch:AVX512)
        endif()
    endif()
    if (ENABLE_AVX2)
        add_compile_definitions(ENABLE_AVX2)
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            message("GNU or Clang compiler, AVX2 optimization ON")
            add_compile_options(-O3 -ftree-vectorize -mavx2)
        elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
            message("Intel compiler, AVX2 optimization ON")
            add_compile_options(-O3 -msse4.2 -axAVX,core-avx2)
            add_compile_options(-axAVX -xAVX)
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            message("MSVC compiler, AVX2 optimization ON")
            add_compile_options(/O2 /arch:AVX2)
        endif()
    endif()
    if (ENABLE_AVX)
        add_compile_definitions(ENABLE_AVX)
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            message("GNU or Clang compiler, AVX optimization ON")
            add_compile_options(-O3 -ftree-vectorize -mavx)
        elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
            message("Intel compiler, AVX optimization ON")
            add_compile_options(-Ofast -axAVX -xAVX)
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            message("MSVC compiler, AVX optimization ON")
            add_compile_options(/O2 /arch:AVX)
        endif()
    endif()
    if (ENABLE_SSE3)
        add_compile_definitions(ENABLE_SSE3)
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
        add_compile_definitions(ENABLE_SSE2)
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
    sklearn
    GIT_REPOSITORY https://github.com/mgorshkov/sklearn.git
    GIT_TAG 0.0.4
)

FetchContent_MakeAvailable(sklearn)

include_directories(${sklearn_SOURCE_DIR}/include)

add_executable(${GMT_TREND_2D})

target_sources(${GMT_TREND_2D} PUBLIC main.cpp)

target_link_libraries(
    ${GMT_TREND_2D}
    pd
    sklearn
    ${PTHREAD})

install(
    TARGETS ${GMT_TREND_2D}
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT ${GMT_TREND_2D}
)
