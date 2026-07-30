# T
op-level CMake for mlir-mr. Library and app are in src/lib and src/appenable_language(CXX)

# Library target lives in src/lib/ following the conquer-opt pattern.
add_subdirectory(src/lib)

# Provides a stable Docker build target while mlir_mr has no source files.
# Once mlir_mr becomes a STATIC library, this dependency builds it normally.
add_custom_target(mlir-mr-smoke
    COMMAND ${CMAKE_COMMAND} -E echo "mlir_mr CMake target is configured"
)
add_dependencies(mlir-mr-smoke mlir_mr)

# Optional: uncomment when metamorphic transforms need CONQuER's TOSA
# quantisation passes or custom dialect directly.
# target_link_libraries(mlir_mr INTERFACE conquer_lib)

# App subdirectory
add_subdirectory(src/app)



