add_library(mlir_mr_lowering STATIC lowering.cpp)
add_library(MLIRMR::mlir_mr_lowering ALIAS mlir_mr_lowering)

set_property(TARGET mlir_mr_lowering PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mr_lowering PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mr_lowering PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(mlir_mr_lowering PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../../include>
)

target_link_libraries(mlir_mr_lowering PUBLIC
    llvm_mlir_headers
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRAffineToStandard
    MLIRSCFToControlFlow
    MLIRArithToLLVM
    MLIRFuncToLLVM
    MLIRLLVMToLLVMIRTranslation
    MLIRMemRefToLLVM
    MLIROpenMPToLLVM
    MLIRControlFlowToLLVM
    MLIRReconcileUnrealizedCasts
)
