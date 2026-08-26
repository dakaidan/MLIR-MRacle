enable_language(CXX)

# Optimisation level for all mlir_mracle targets. ThreadSanitizer is the only
# sanitizer: the JIT instruments JIT'd code with TSan and needs the runtime
# linked into the host binary at startup, so the tool is always built with
# -fsanitize=thread.
set(MLIR_MRACLE_OPT_LEVEL "2" CACHE STRING
    "Optimisation level for mlir_mracle targets")
add_compile_options(-fsanitize=thread -O${MLIR_MRACLE_OPT_LEVEL})
add_link_options(-fsanitize=thread)

# Apple clang does not auto-detect Homebrew's libomp; hint FindOpenMP when
# present. Done here so both the oracle library and the app resolve the same
# OpenMP runtime.
if(APPLE AND NOT OpenMP_ROOT)
    foreach(dir /opt/homebrew/opt/libomp /usr/local/opt/libomp)
        if(EXISTS "${dir}/lib" AND EXISTS "${dir}/include/omp.h")
            set(OpenMP_ROOT "${dir}")
            break()
        endif()
    endforeach()
endif()
find_package(OpenMP REQUIRED)

add_subdirectory(src/lib)
add_subdirectory(src/app)

# Provides a stable Docker build target while mlir_mracle has no source files.
# Once mlir_mracle becomes a STATIC library, this dependency builds it normally.
add_custom_target(mlir_mracle-smoke
    COMMAND ${CMAKE_COMMAND} -E echo "mlir_mracle CMake target is configured"
)
add_dependencies(mlir_mracle-smoke mlir_mracle)

# Optional: uncomment when metamorphic transforms need CONQuER's TOSA
# quantisation passes or custom dialect directly.
# target_link_libraries(mlir_mracle INTERFACE conquer_lib)

target_include_directories(mlir_mracle INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

