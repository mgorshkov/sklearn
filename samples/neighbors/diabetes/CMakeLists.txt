cmake_minimum_required(VERSION 3.13.0)

set(NEIGHBORS_DIABETES neighbors_diabetes)

project(${NEIGHBORS_DIABETES})

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)

FetchContent_Declare(
    sklearn
    GIT_REPOSITORY https://github.com/mgorshkov/sklearn.git
    GIT_TAG 0.0.2
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

add_executable(${NEIGHBORS_DIABETES})

target_sources(${NEIGHBORS_DIABETES} PUBLIC main.cpp)

target_link_libraries(
    ${NEIGHBORS_DIABETES}
    pd
    ssl
    sklearn
    ${PTHREAD})

install(
    TARGETS ${NEIGHBORS_DIABETES}
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT ${NEIGHBORS_DIABETES}
)
