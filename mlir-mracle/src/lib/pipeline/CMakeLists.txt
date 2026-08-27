add_library(mlir_mracle_pipeline STATIC
    pipeline.cpp
    pipeline_common.cpp
    emit/emit_pipeline.cpp
    execute/execution_pipeline.cpp
)
add_library(MLIRMRACLE::mlir_mracle_pipeline ALIAS mlir_mracle_pipeline)

set_property(TARGET mlir_mracle_pipeline PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mracle_pipeline PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mracle_pipeline PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(mlir_mracle_pipeline PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
)

find_package(OpenMP REQUIRED)

target_link_libraries(mlir_mracle_pipeline PUBLIC
    mlir_mracle_execution
    mlir_mracle_oracle
    mlir_mracle_io
    mlir_mracle_backend
    mlir_mracle_passes
    mlir_mracle_core
    mlir_mracle_context
    llvm_mlir_headers
    MLIRIR
    MLIRParser
    LLVMCore
    LLVMSupport
    OpenMP::OpenMP_CXX
)
