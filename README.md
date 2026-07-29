# MLIR Metamorphic Testing

Development environment for metamorphic testing of MLIR compiler infrastructure, with CONQuER's TOSA quantisation system.

## Repository layout

- `conquer-opt/`: the existing C++ MLIR/TOSA quantisation compiler and the pinned IREE/LLVM/MLIR dependency setup.
- `mlir-mr/`: an initially empty, MLIR-linked CMake library for reusable metamorphic transforms and test infrastructure.
- `fuzzing/`: scratch space and conventions for generators, AFL++ harnesses, corpora, findings, and reduced failures.

## Setup

Clone with submodules, initialise IREE's direct dependencies, then build:

```bash
git submodule update --init
git -C conquer-opt/third_party/iree submodule update --init
docker build -t mlir-metamorphic-testing .
docker run -it --rm mlir-metamorphic-testing
```

The initial build is large because it compiles the pinned LLVM/MLIR and IREE
stack. Docker caching makes subsequent source changes much cheaper.

To configure without Docker:

```bash
cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCONQUER_ENABLE_IREE=ON \
  -DCONQUER_BUILD_TESTS=OFF
cmake --build build --target conquer-opt
```

See `mlir-mr/README.md` for the library boundary and `fuzzing/README.md` for
the experimental-tooling workflow.

## Docker build modes

Running the image without a command executes the smoke test and exits:

```bash
docker run --rm mlir-metamorphic-testing
```

Build only the `mlir_mr` target and shared MLIR/IREE dependencies:

```bash
docker build --build-arg BUILD_CONQUER_OPT=OFF -t mlir-metamorphic-testing .
```

Use an interactive shell instead of the default smoke test with:

```bash
docker run -it --rm mlir-metamorphic-testing /bin/bash
```
