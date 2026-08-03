add_subdirectory(jit)
add_subdirectory(lowering)
add_library(mlir_mr_backend INTERFACE)
add_library(MLIRMR::mlir_mr_backend ALIAS mlir_mr_backend)

target_include_directories(mlir_mr_backend INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
)

target_link_libraries(mlir_mr_backend INTERFACE
    mlir_mr_jit
    mlir_mr_lowering
)
