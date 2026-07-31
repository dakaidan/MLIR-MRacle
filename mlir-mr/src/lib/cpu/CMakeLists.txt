add_library(mlir_mr_cpu INTERFACE)
add_library(MLIRMR::mlir_mr_cpu ALIAS mlir_mr_cpu)

set_property(TARGET mlir_mr_cpu PROPERTY CXX_STANDARD 23)
set_property(TARGET mlir_mr_cpu PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET mlir_mr_cpu PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(mlir_mr_cpu INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
    ${MLIR_INCLUDE_DIRS}
)

# TODO: Add CPU-specific deps here when first .cpp.
# TODO: Switch add_library to STATIC at the same time.
