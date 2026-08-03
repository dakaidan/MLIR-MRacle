add_library(mlir_mr_io STATIC io.cpp)
add_library(MLIRMR::mlir_mr_io ALIAS mlir_mr_io)

set_property(TARGET mlir_mr_io PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mr_io PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mr_io PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(mlir_mr_io PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
)

target_link_libraries(mlir_mr_io PUBLIC
    llvm_mlir_headers
    LLVMCore
    LLVMSupport
    MLIRIR
    MLIRBuiltinToLLVMIRTranslation
    MLIRLLVMToLLVMIRTranslation
)
