file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR} "datasets/*.cpp" "metrics/*.cpp" "model_selection/*.cpp" "preprocessing/*.cpp")

add_library(${SKLEARN} ${SOURCES})

target_include_directories(${SKLEARN} PUBLIC ${np_SOURCE_DIR}/include ${pd_SOURCE_DIR}/include ${scipy_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include)