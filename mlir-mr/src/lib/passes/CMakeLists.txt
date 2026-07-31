add_library(mlir_mr_passes INTERFACE)
add_library(MLIRMR::mlir_mr_passes ALIAS mlir_mr_passes)

set_property(TARGET mlir_mr_passes PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mr_passes PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mr_passes PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(mlir_mr_passes INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
    ${MLIR_INCLUDE_DIRS}
)

# TODO: Add pass-specific deps here when first .cpp.
# TODO: Switch add_library to STATIC at the same time.
