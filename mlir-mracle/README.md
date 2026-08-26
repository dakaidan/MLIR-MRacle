# mlir-mracle

The C++ library root for the MLIR-MRacle project. It provides MLIR parsing,
the metamorphic transform registry, pipeline orchestration, JIT lowering and
execution, and the outcome oracle.

`mlir_mracle` is a CMake INTERFACE target that aggregates the per-module
libraries under `src/lib/`; the `mlir_mracle_opt` CLI in `src/app/` is the
only consumer today. See the repository root `README.md` for usage.

