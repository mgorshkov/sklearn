cmake_minimum_required(VERSION 3.13.0)

set(SKLEARN_UNIT_TESTS sklearn_unit_tests)

project(${SKLEARN_UNIT_TESTS} VERSION 0.0.1 DESCRIPTION "Sklearn Unit Tests")

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if (NOT DEFINED CMAKE_INSTALL_BINDIR)
    set(CMAKE_INSTALL_BINDIR "bin" CACHE PATH "user executables (bin)")
endif()

if (MSVC)
    add_compile_options(/W4 /WX)
	add_definitions(-DNOMINMAX) # allow using min/max in the project
else()
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG main
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(BUILD_GTEST ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

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

include_directories(../include include ${googletest_SOURCE_DIR}/googletest/include ${np_SOURCE_DIR}/include ${scipy_SOURCE_DIR}/include)

add_executable(${SKLEARN_UNIT_TESTS})

file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR} "src/*.cpp")
target_sources(${SKLEARN_UNIT_TESTS} PUBLIC ${SOURCES})

add_test(NAME ${SKLEARN_UNIT_TESTS} COMMAND tests)

target_link_libraries(
    ${SKLEARN_UNIT_TESTS}
    sklearn
    gtest
    ${PTHREAD})

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${SKLEARN_UNIT_TESTS}
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT ${SKLEARN_UNIT_TESTS}
)