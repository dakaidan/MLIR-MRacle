cmake_minimum_required(VERSION 3.20)

project(MLIRMetamorphicTestingSuperbuild LANGUAGES NONE)

# ConQUER owns the pinned IREE/LLVM/MLIR setup. Adding it first makes those
# CMake targets available to the independent metamorphic-testing subproject.
add_subdirectory(conquer-opt)
add_subdirectory(mlir-mr)
