# Fuzzing and external-tool workspace

Keep metamorphic transforms in `mlir-mr/`; use this area to connect your metamorphic system with fuzzers.

## External tools

Generators such as MLIR-Forge, FLEX, grammar-based generators, and other tools can be:

- added as a pinned Git submodule beneath `fuzzing/tools/<name>` when the project should consume an exact upstream revision; or
- copied into `fuzzing/tools/<name>` and developed within this git tree when modifications are needed.

