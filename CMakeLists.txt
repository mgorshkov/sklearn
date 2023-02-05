cmake_minimum_required(VERSION 3.13.0)

set(SKLEARN sklearn)

project(${SKLEARN} VERSION 0.0.1 DESCRIPTION "ML Methods from scikit-learn library")

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if (MSVC)
    add_compile_options(/W4 /WX)
    add_definitions(-DNOMINMAX) # allow using min/max in the SKLEARN
else()
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

include(FetchContent)

FetchContent_Declare(
    np
    URL https://mgorshkov.jfrog.io/artifactory/default-generic-local/np/np-0.0.4.tgz
)

FetchContent_MakeAvailable(np)

FetchContent_Declare(
    scipy
    URL https://mgorshkov.jfrog.io/artifactory/default-generic-local/scipy/scipy-0.0.1.tgz
)

FetchContent_MakeAvailable(scipy)

include_directories(include ${np_SOURCE_DIR}/include ${scipy_SOURCE_DIR}/include)

file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR} "src/datasets/*.cpp")
add_library(${SKLEARN} ${SOURCES})

# Make unit tests and doc if it's the main project
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(MAKE_UNIT_TESTS ON)
    set(MAKE_DOC ON)
    set(MAKE_SAMPLE ON)
else()
    set(MAKE_UNIT_TESTS OFF)
    set(MAKE_DOC OFF)
    set(MAKE_SAMPLE OFF)
endif()

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

if (MAKE_SAMPLE)
    message(STATUS "Enable samples build in ${SKLEARN}")
    add_subdirectory(samples/neighbors)
endif()

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/sklearn
    DESTINATION include
    COMPONENT ${SKLEARN_UNIT_TESTS}
)
