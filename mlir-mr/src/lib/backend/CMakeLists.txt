add_library(mlir_mr_backend INTERFACE)
add_library(MLIRMR::mlir_mr_backend ALIAS mlir_mr_backend)

set_property(TARGET mlir_mr_backend PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mr_backend PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mr_backend PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(mlir_mr_backend INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    ${MLIR_INCLUDE_DIRS}
)

# TODO: Add backend-specific deps here when first .cpp.
# TODO: Switch add_library to STATIC at the same time.
