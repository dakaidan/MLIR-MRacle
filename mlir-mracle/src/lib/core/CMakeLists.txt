add_library(mlir_mracle_core STATIC core.cpp)
add_library(MLIRMRACLE::mlir_mracle_core ALIAS mlir_mracle_core)

set_property(TARGET mlir_mracle_core PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mracle_core PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mracle_core PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(mlir_mracle_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
)

target_link_libraries(mlir_mracle_core PUBLIC
    mlir_mracle_io
    mlir_mracle_context
    mlir_mracle_oracle
    mlir_mracle_execution
    mlir_mracle_lowering
    mlir_mracle_jit
    llvm_mlir_headers
    MLIRIR
    MLIRParser
    LLVMSupport
    LLVMBitReader
    LLVMBitWriter
)
