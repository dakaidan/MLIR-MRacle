add_subdirectory(backend)
add_subdirectory(passes)
add_subdirectory(context)
add_subdirectory(io)

add_library(mlir_mr INTERFACE)
add_library(MLIRMR::mlir_mr ALIAS mlir_mr)

set_property(TARGET mlir_mr PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mr PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mr PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(mlir_mr INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
)

target_link_libraries(mlir_mr INTERFACE
    mlir_mr_backend
    mlir_mr_passes
    mlir_mr_context
    mlir_mr_io
)

