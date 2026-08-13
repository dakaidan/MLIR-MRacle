enable_language(CXX)

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

# Provides a stable Docker build target while mlir_mr has no source files.
# Once mlir_mr becomes a STATIC library, this dependency builds it normally.
add_custom_target(mlir-mr-smoke
    COMMAND ${CMAKE_COMMAND} -E echo "mlir_mr CMake target is configured"
)
add_dependencies(mlir-mr-smoke mlir_mr)

# Optional: uncomment when metamorphic transforms need CONQuER's TOSA
# quantisation passes or custom dialect directly.
# target_link_libraries(mlir_mr INTERFACE conquer_lib)

# Must set MLIR_TABLEGEN_EXE and MLIR_TABLEGEN_TARGET to local vendored build
# for MLIR to behave prperly and for mlir-mr to build properly.
set(MLIR_TABLEGEN_EXE ${CMAKE_BINARY_DIR}/llvm_project_build/bin/mlir-tblgen)
set(MLIR_TABLEGEN_TARGET mlir-tblgen)

target_include_directories(mlir_mr INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

