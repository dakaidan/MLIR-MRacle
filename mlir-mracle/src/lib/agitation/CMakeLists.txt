add_library(mlir_mracle_agitation STATIC agitation.cpp)
add_library(MLIRMRACLE::mlir_mracle_agitation ALIAS mlir_mracle_agitation)

set_property(TARGET mlir_mracle_agitation PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mracle_agitation PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mracle_agitation PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(mlir_mracle_agitation PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
)

target_link_libraries(mlir_mracle_agitation PUBLIC
    mlir_mracle_jit
    mlir_mracle_context
    llvm_mlir_headers
    LLVMCore
    LLVMSupport
    LLVMTransformUtils
)
