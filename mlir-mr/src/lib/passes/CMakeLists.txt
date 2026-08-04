set(LLVM_TARGET_DEFINITIONS
${CMAKE_CURRENT_SOURCE_DIR}/../../../include/mlir-mr/passes/MetamorphicMemoryModelPass.td)

get_target_property(MLIRMR_TABLEGEN_INCLUDES llvm_mlir_headers INTERFACE_INCLUDE_DIRECTORIES)
include_directories(${MLIRMR_TABLEGEN_INCLUDES})

mlir_tablegen(MetamorphicMemoryModelPass.inc -gen-pass-decls -name MetamorphicMemoryModel)
add_public_tablegen_target(MLIRMRMetamorphicMemoryModelPassIncGen)

add_library(mlir_mr_passes STATIC MetamorphicMemoryModelPass.cpp)
add_library(MLIRMR::mlir_mr_passes ALIAS mlir_mr_passes)
target_compile_options(mlir_mr_passes PRIVATE -fno-rtti)

set_property(TARGET mlir_mr_passes PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mr_passes PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mr_passes PROPERTY CXX_EXTENSIONS OFF)

add_dependencies(mlir_mr_passes MLIRMRMetamorphicMemoryModelPassIncGen)

target_include_directories(mlir_mr_passes PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

target_link_libraries(mlir_mr_passes PUBLIC
    llvm_mlir_headers
    MLIRIR
    MLIRPass
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRLLVMDialect
)