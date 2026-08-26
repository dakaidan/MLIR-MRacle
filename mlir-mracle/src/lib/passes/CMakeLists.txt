set(LLVM_TARGET_DEFINITIONS
${CMAKE_CURRENT_SOURCE_DIR}/../../../include/mlir-mracle/passes/MetamorphicPass.td)

get_target_property(MLIRMRACLE_TABLEGEN_INCLUDES llvm_mlir_headers INTERFACE_INCLUDE_DIRECTORIES)
include_directories(${MLIRMRACLE_TABLEGEN_INCLUDES})

mlir_tablegen(MetamorphicPass.inc -gen-pass-decls -name Metamorphic)
add_public_tablegen_target(MLIRMRACLEMetamorphicPassIncGen)

add_subdirectory(transforms)

add_library(mlir_mracle_passes STATIC MetamorphicPass.cpp)
add_library(MLIRMRACLE::mlir_mracle_passes ALIAS mlir_mracle_passes)
target_compile_options(mlir_mracle_passes PRIVATE -fno-rtti)

set_property(TARGET mlir_mracle_passes PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mracle_passes PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mracle_passes PROPERTY CXX_EXTENSIONS OFF)

add_dependencies(mlir_mracle_passes MLIRMRACLEMetamorphicPassIncGen)

target_include_directories(mlir_mracle_passes PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

target_link_libraries(mlir_mracle_passes PUBLIC
    mlir_mracle_transforms
    llvm_mlir_headers
    MLIRIR
    MLIRPass
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRLLVMDialect
    MLIROpenMPDialect
)