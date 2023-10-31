cmake_minimum_required(VERSION 3.13.0)

set(PROJECT_NAME sklearn_unit_tests)

project(${PROJECT_NAME} VERSION 0.0.4 DESCRIPTION "Sklearn Unit Tests")

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

set(OPENSSL ON)

add_executable(${PROJECT_NAME})

file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR} "src/*.cpp")
target_sources(${PROJECT_NAME} PUBLIC ${SOURCES})

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ../include
        include
        ${np_SOURCE_DIR}/include
        ${scipy_SOURCE_DIR}/include
)

target_include_directories(${PROJECT_NAME}
    SYSTEM PRIVATE
        ${googletest_SOURCE_DIR}/googletest/include
)

add_test(NAME ${PROJECT_NAME} COMMAND tests)

set (LIBS
    ${LIBS}
    pd
    ssl
    sklearn
    gtest
    pthread)

# Apply target-dependent cmake options
include(${cmake_SOURCE_DIR}/Target.cmake)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT ${PROJECT_NAME}
)
