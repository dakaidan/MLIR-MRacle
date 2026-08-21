#pragma once

#include "llvm/IR/Module.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace mlir_mracle {

// OpenMP settings for one agitation config. The team size is the only knob
// that affects omp.sections-based tests: schedule/chunk only apply to
// worksharing loops, and runtime dynamic team adjustment is disabled so
// omp_set_num_threads stays authoritative.
struct OpenMPSettings {
    int numThreads = 2;
};

struct AgitationConfig {
    OpenMPSettings omp;
    int runs = 0; // assigned by the experiment runner
};

struct CompileOptions {
    // Number of distinct in-memory binaries to compile per side; each gets a
    // CodeGen opt level from {0, 1, 2, 3} and its own basic-block layout
    // derived from shuffleSeed. With binaryCount >= 4 every level is
    // guaranteed to appear at least once; the order is seeded so source and
    // transformed sides compile with the same sequence.
    int binaryCount = 5;
    bool shuffleCode = true;
    uint32_t shuffleSeed = 42;
    bool enableTsan = false;
};

struct CompiledBinary {
    std::string side; // "source" | "transformed"
    int compileIndex = 0;
    int jitOptLevel = -1;
    std::function<std::vector<int64_t>()> fn;
};

// Generates configCount unique random thread counts from {2,3,4,6,8}; the
// order is randomised from the seed. Runs are assigned by the experiment
// runner.
std::vector<AgitationConfig> generateAgitationConfigs(uint32_t seed,
                                                      int configCount);

// Compiles the module once per binaryCount entry. When shuffleCode is set,
// each variant's basic blocks are shuffled (entry stays first) and every
// basic block is emitted into its own section so the shuffled order is kept
// in the generated machine code. Each variant uses a random opt level from
// {0, 1, 2, 3} drawn from shuffleSeed.
std::vector<CompiledBinary> compileBinarySet(const llvm::Module &module,
                                             const CompileOptions &opts,
                                             std::string side,
                                             std::string *error = nullptr);

} // namespace mlir_mracle
