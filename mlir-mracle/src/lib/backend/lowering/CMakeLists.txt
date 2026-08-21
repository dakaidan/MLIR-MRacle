add_library(mlir_mracle_lowering STATIC lowering.cpp)
add_library(MLIRMRACLE::mlir_mracle_lowering ALIAS mlir_mracle_lowering)

set_property(TARGET mlir_mracle_lowering PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mracle_lowering PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mracle_lowering PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(mlir_mracle_lowering PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../../include>
)

target_link_libraries(mlir_mracle_lowering PUBLIC
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
