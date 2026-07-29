# This starts as a header/source-free library so students can choose the public
# API and directory layout in their MRs without deleting placeholder code.
add_library(mlir_mr INTERFACE)
add_library(MLIRMR::mlir_mr ALIAS mlir_mr)

target_compile_features(mlir_mr INTERFACE cxx_std_23)

target_include_directories(mlir_mr INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# These targets come from the pinned LLVM/MLIR checkout configured by
# conquer-opt. Add narrower dialect/transform libraries as implementations
# require them.
target_link_libraries(mlir_mr INTERFACE
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRRewrite
    MLIRSupport
    MLIRTransforms
    MLIRTosaDialect
)

# Provides a stable Docker build target while mlir_mr has no source files.
# Once mlir_mr becomes a STATIC library, this dependency builds it normally.
add_custom_target(mlir-mr-smoke
    COMMAND ${CMAKE_COMMAND} -E echo "mlir_mr CMake target is configured"
)
add_dependencies(mlir-mr-smoke mlir_mr)

# Optional: uncomment when metamorphic transforms need CONQuER's TOSA
# quantisation passes or custom dialect directly.
# target_link_libraries(mlir_mr INTERFACE conquer_lib)
