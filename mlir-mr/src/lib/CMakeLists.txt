# Subdirectory added here + sub-libraries below
add_subdirectory(backend)
add_subdirectory(cpu)
add_subdirectory(passes)

add_library(mlir_mr INTERFACE)
add_library(MLIRMR::mlir_mr ALIAS mlir_mr)

set_property(TARGET mlir_mr PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mr PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mr PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(mlir_mr INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    ${MLIR_INCLUDE_DIRS}
)

# Add sub-library deps here when creating new folders
target_link_libraries(mlir_mr INTERFACE
    mlir_mr_backend
    mlir_mr_cpu
    mlir_mr_passes
)

# Shared MLIR/LLVM deps
# TODO: Migrate each to the lowest sub-library that needs it once that sub-library has its first .cpp source
target_link_libraries(mlir_mr INTERFACE
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRRewrite
    MLIRSupport
    MLIRTransforms
    MLIRTosaDialect
    MLIRArithDialect
    LLVMCore
    LLVMSupport
    LLVMBinaryFormat
    LLVMAsmPrinter
    LLVMSupport
    MLIRToLLVMIRTranslationRegistration
    MLIRAffineToStandard
    MLIRControlFlowToLLVM
    MLIRFuncToLLVM
    MLIRArithToLLVM
    MLIRSCFToControlFlow
    MLIRRegisterAllDialects
    LLVMOrcJIT
    LLVMJITLink
    LLVMOrcTargetProcess
    LLVMExecutionEngine
)
