find_package(OpenMP REQUIRED)

add_library(mlir_mracle_execution STATIC
    execution.cpp
)
add_library(MLIRMRACLE::mlir_mracle_execution ALIAS mlir_mracle_execution)

set_property(TARGET mlir_mracle_execution PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mracle_execution PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mracle_execution PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(mlir_mracle_execution PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
)

target_link_libraries(mlir_mracle_execution PUBLIC
    mlir_mracle_agitation
    mlir_mracle_oracle
    OpenMP::OpenMP_CXX
)