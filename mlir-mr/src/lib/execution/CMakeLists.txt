find_package(OpenMP REQUIRED)

add_library(mlir_mr_execution STATIC
    execution.cpp
)
add_library(MLIRMR::mlir_mr_execution ALIAS mlir_mr_execution)

set_property(TARGET mlir_mr_execution PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mr_execution PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mr_execution PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(mlir_mr_execution PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
)

target_link_libraries(mlir_mr_execution PUBLIC
    mlir_mr_core
    mlir_mr_agitation
    mlir_mr_oracle
    OpenMP::OpenMP_CXX
)