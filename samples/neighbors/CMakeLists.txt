cmake_minimum_required(VERSION 3.13.0)

set(PROJECT neighbors)

project(${PROJECT})

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if (MSVC)
    add_compile_options(/W4 /WX)
	add_definitions(-DNOMINMAX) # allow using min/max in the project
else()
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

include_directories(../../include)

add_executable(${PROJECT})

target_sources(${PROJECT} PUBLIC main.cpp)

target_link_libraries(
    ${PROJECT}
    sklearn
    ${PTHREAD})

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT}
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT ${PROJECT}
)
