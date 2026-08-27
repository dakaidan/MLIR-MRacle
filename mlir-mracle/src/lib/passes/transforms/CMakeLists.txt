add_library(mlir_mracle_transforms STATIC
    generic/GenericTransforms.cpp
    arm/ArmTransforms.cpp
    MetamorphicTransform.cpp
)
add_library(MLIRMRACLE::mlir_mracle_transforms ALIAS mlir_mracle_transforms)

target_compile_options(mlir_mracle_transforms PRIVATE -fno-rtti)

set_property(TARGET mlir_mracle_transforms PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mracle_transforms PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mracle_transforms PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(mlir_mracle_transforms PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../../include>
)

target_link_libraries(mlir_mracle_transforms PUBLIC
    mlir_mracle_core
    llvm_mlir_headers
    MLIRIR
    MLIRFuncDialect
    MLIRArithDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIROpenMPDialect
)
