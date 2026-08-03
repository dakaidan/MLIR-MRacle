cmake_minimum_required(VERSION 3.20)

project(MLIRMetamorphicTestingSuperbuild LANGUAGES NONE)

set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "")
set(LLVM_TARGETS_TO_BUILD "X86;AArch64" CACHE STRING "")  
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")

add_subdirectory(
    ${CMAKE_SOURCE_DIR}/llvm-project/llvm    # source
    ${CMAKE_BINARY_DIR}/llvm_project_build   # separate binary dir
)

add_library(llvm_mlir_headers INTERFACE)
target_include_directories(llvm_mlir_headers INTERFACE
    ${CMAKE_SOURCE_DIR}/llvm-project/llvm/include
    ${CMAKE_BINARY_DIR}/llvm_project_build/include
    ${CMAKE_SOURCE_DIR}/llvm-project/mlir/include
    ${CMAKE_BINARY_DIR}/llvm_project_build/tools/mlir/include
)

# ConQUER owns the pinned IREE/LLVM/MLIR setup. Adding it first makes those
# CMake targets available to the independent metamorphic-testing subproject.
# add_subdirectory(conquer-opt)
add_subdirectory(mlir-mr)
