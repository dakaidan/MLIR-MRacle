enable_language(CXX)

# Sanitizer and optimisation configuration for all mlir_mracle targets. The
# default (TSan at -O2) keeps race detection for JIT'd programs while keeping
# the tool itself fast; TSan is already applied explicitly to JIT'd code.
# "address,undefined" builds an ASan+UBSan tool that catches parser/lowering/
# JIT memory and UB bugs. ASan and TSan must never be combined in one build;
# an empty value disables sanitizers entirely.
set(MLIR_MRACLE_SANITIZERS "thread" CACHE STRING
    "Sanitizers for mlir_mracle targets (comma-separated; empty disables)")
set(MLIR_MRACLE_OPT_LEVEL "2" CACHE STRING
    "Optimisation level for mlir_mracle targets")
if(NOT MLIR_MRACLE_SANITIZERS STREQUAL "")
    add_compile_options(-fsanitize=${MLIR_MRACLE_SANITIZERS} -O${MLIR_MRACLE_OPT_LEVEL})
    add_link_options(-fsanitize=${MLIR_MRACLE_SANITIZERS})
else()
    add_compile_options(-O${MLIR_MRACLE_OPT_LEVEL})
endif()

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

# Must set MLIR_TABLEGEN_EXE and MLIR_TABLEGEN_TARGET to local vendored build
# for MLIR to behave prperly and for mlir_mracle to build properly.
set(MLIR_TABLEGEN_EXE ${CMAKE_BINARY_DIR}/llvm_project_build/bin/mlir-tblgen)
set(MLIR_TABLEGEN_TARGET mlir-tblgen)

target_include_directories(mlir_mracle INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

