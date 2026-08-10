add_library(mlir_mr_context STATIC context.cpp)
add_library(MLIRMR::mlir_mr_context ALIAS mlir_mr_context)

set_property(TARGET mlir_mr_context PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mr_context PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mr_context PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(mlir_mr_context PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
)

target_link_libraries(mlir_mr_context PUBLIC
    llvm_mlir_headers
    MLIRIR
    MLIRPass
    MLIRRegisterAllDialects
    MLIRBuiltinToLLVMIRTranslation
    MLIRLLVMToLLVMIRTranslation
    MLIROpenMPToLLVMIRTranslation
)
