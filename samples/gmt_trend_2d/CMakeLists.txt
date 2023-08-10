cmake_minimum_required(VERSION 3.13.0)

set(GMT_TREND_2D gmt_trend_2d)

project(${GMT_TREND_2D})

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)

FetchContent_Declare(
    sklearn
    GIT_REPOSITORY https://github.com/mgorshkov/sklearn.git
    GIT_TAG main
)

FetchContent_MakeAvailable(sklearn)

find_package(OpenMP)
if (OPENMP_FOUND)
    add_definitions(-DOPENMP)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(${sklearn_SOURCE_DIR}/include)

add_executable(${GMT_TREND_2D})

target_sources(${GMT_TREND_2D} PUBLIC main.cpp)

target_link_libraries(
    ${GMT_TREND_2D}
    pd
    ssl
    sklearn
    ${PTHREAD})

install(
    TARGETS ${GMT_TREND_2D}
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT ${GMT_TREND_2D}
)
