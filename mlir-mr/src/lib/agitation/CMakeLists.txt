add_library(mlir_mr_agitation STATIC agitation.cpp)
add_library(MLIRMR::mlir_mr_agitation ALIAS mlir_mr_agitation)

set_property(TARGET mlir_mr_agitation PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mr_agitation PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mr_agitation PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(mlir_mr_agitation PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
)

target_link_libraries(mlir_mr_agitation PUBLIC
    mlir_mr_jit
    mlir_mr_context
    llvm_mlir_headers
    LLVMCore
    LLVMSupport
    LLVMTransformUtils
)
