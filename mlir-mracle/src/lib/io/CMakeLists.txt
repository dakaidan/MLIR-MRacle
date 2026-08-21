add_library(mlir_mracle_io STATIC io.cpp)
add_library(MLIRMRACLE::mlir_mracle_io ALIAS mlir_mracle_io)

set_property(TARGET mlir_mracle_io PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mracle_io PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mracle_io PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(mlir_mracle_io PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
)

target_link_libraries(mlir_mracle_io PUBLIC
    mlir_mracle_context
    llvm_mlir_headers
    LLVMCore
    LLVMSupport
    MLIRIR
    MLIRBuiltinToLLVMIRTranslation
    MLIRLLVMToLLVMIRTranslation
)
