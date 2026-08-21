add_subdirectory(jit)
add_subdirectory(lowering)
add_library(mlir_mracle_backend INTERFACE)
add_library(MLIRMRACLE::mlir_mracle_backend ALIAS mlir_mracle_backend)

target_include_directories(mlir_mracle_backend INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
)

target_link_libraries(mlir_mracle_backend INTERFACE
    mlir_mracle_jit
    mlir_mracle_lowering
)
