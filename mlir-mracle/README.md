# MLIR metamorphic-testing library

This is the home for the reusable C++ library that will parse MLIR, construct metamorphic transformations, run pass pipelines, and support test oracles.

The initial target is an empty CMake INTERFACE library. 
Add headers under `include/mlir-mracle/`, implementations under `lib/`, and tests under `tests/`. When the first implementation is added, change `mlir_mracle` from an INTERFACE library to a normal STATIC library and adjust the CMake visibility keywords.

The target already inherits the pinned MLIR core, parser, pass, rewrite, transform, and TOSA libraries. IREE is configured by the parent build and its targets are available if an oracle later needs compilation or execution. The commented `conquer_lib` link enables direct reuse of CONQuER's quantisation passes when required.
