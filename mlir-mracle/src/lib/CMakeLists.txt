add_subdirectory(core)
add_subdirectory(backend)
add_subdirectory(passes)
add_subdirectory(context)
add_subdirectory(oracle)
add_subdirectory(agitation)
add_subdirectory(execution)
add_subdirectory(io)
add_subdirectory(pipeline)
add_subdirectory(legacy)

add_library(mlir_mracle INTERFACE)
add_library(MLIRMRACLE::mlir_mracle ALIAS mlir_mracle)

set_property(TARGET mlir_mracle PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mracle PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mracle PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(mlir_mracle INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
)

target_link_libraries(mlir_mracle INTERFACE
    mlir_mracle_core
    mlir_mracle_backend
    mlir_mracle_passes
    mlir_mracle_context
    mlir_mracle_oracle
    mlir_mracle_agitation
    mlir_mracle_execution
    mlir_mracle_io
    mlir_mracle_pipeline
    mlir_mracle_legacy
)

