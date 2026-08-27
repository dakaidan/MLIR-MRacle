add_library(mlir_mracle_legacy STATIC
    legacy_oracle.cpp
    legacy_pipeline.cpp
    source_memo.cpp
)
add_library(MLIRMRACLE::mlir_mracle_legacy ALIAS mlir_mracle_legacy)

set_property(TARGET mlir_mracle_legacy PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mracle_legacy PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mracle_legacy PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(mlir_mracle_legacy PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
)

find_package(OpenMP REQUIRED)

target_link_libraries(mlir_mracle_legacy PUBLIC
    mlir_mracle_pipeline
    mlir_mracle_execution
    mlir_mracle_io
    mlir_mracle_jit
    mlir_mracle_core
    mlir_mracle_context
    llvm_mlir_headers
    LLVMCore
    LLVMSupport
    LLVMBitWriter
    LLVMTransformUtils
    OpenMP::OpenMP_CXX
)

# regression tests for the legacy oracle judges
add_executable(mlir_mracle_legacy_oracle_test legacy_oracle_test.cpp)

set_property(TARGET mlir_mracle_legacy_oracle_test PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mracle_legacy_oracle_test PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mracle_legacy_oracle_test PROPERTY CXX_EXTENSIONS OFF)

target_link_libraries(mlir_mracle_legacy_oracle_test PRIVATE MLIRMRACLE::mlir_mracle_legacy)

add_test(NAME legacy_oracle_test COMMAND mlir_mracle_legacy_oracle_test)
