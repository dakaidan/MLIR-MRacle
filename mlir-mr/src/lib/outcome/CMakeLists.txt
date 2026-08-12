add_library(mlir_mr_outcome STATIC outcome.cpp)
add_library(MLIRMR::mlir_mr_outcome ALIAS mlir_mr_outcome)

set_property(TARGET mlir_mr_outcome PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mr_outcome PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mr_outcome PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(mlir_mr_outcome PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
)

find_package(Threads REQUIRED)

target_link_libraries(mlir_mr_outcome PUBLIC
    Threads::Threads
)
