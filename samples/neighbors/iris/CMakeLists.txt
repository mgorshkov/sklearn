cmake_minimum_required(VERSION 3.13.0)

set(NEIGHBORS_IRIS neighbors_iris)

project(${NEIGHBORS_IRIS})

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)

FetchContent_Declare(
    sklearn
    GIT_REPOSITORY https://github.com/mgorshkov/sklearn.git
    GIT_TAG main
)

FetchContent_MakeAvailable(sklearn)

include_directories(${sklearn_SOURCE_DIR}/include)

add_executable(${NEIGHBORS_IRIS})

target_sources(${NEIGHBORS_IRIS} PUBLIC main.cpp)

target_link_libraries(
    ${NEIGHBORS_IRIS}
    pd
    ssl
    sklearn
    ${PTHREAD})

install(
    TARGETS ${NEIGHBORS_IRIS}
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT ${NEIGHBORS_IRIS}
)
