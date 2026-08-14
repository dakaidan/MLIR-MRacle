add_library(mlir_mr_core STATIC core.cpp)
add_library(MLIRMR::mlir_mr_core ALIAS mlir_mr_core)

set_property(TARGET mlir_mr_core PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mr_core PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mr_core PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(mlir_mr_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
)

target_link_libraries(mlir_mr_core PUBLIC
    mlir_mr_io
    mlir_mr_context
    mlir_mr_oracle
    mlir_mr_lowering
    mlir_mr_jit
    llvm_mlir_headers
    MLIRIR
    MLIRParser
    LLVMSupport
    LLVMBitReader
    LLVMBitWriter
)
