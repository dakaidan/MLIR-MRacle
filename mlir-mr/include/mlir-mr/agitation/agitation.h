#pragma once

#include "llvm/IR/Module.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace mlir_mr {

enum class ScheduleKind { Static, Dynamic, Guided, Auto };

struct OpenMPSettings {
    int numThreads = 2;
    ScheduleKind schedule = ScheduleKind::Static;
    int chunkSize = 1;
    bool dynamic = false;
};

struct AgitationConfig {
    OpenMPSettings omp;
    int runs = 0; // assigned by the experiment runner
};

struct CompileOptions {
    // Per-compile CodeGen opt levels; empty means {0, 1, 2, 3, 3}.
    std::vector<int> jitOptLevels;
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

// Generates configCount unique random configs (threads from {2,3,4,6,8},
// schedule, chunk and dynamic flag; configs that differ only in thread count
// are still distinct). Runs are assigned by the experiment runner.
std::vector<AgitationConfig> generateAgitationConfigs(uint32_t seed,
                                                      int configCount);

// Compiles the module once per jitOptLevels entry. When shuffleCode is set,
// each variant's basic blocks are shuffled (entry stays first) and every
// basic block is emitted into its own section so the shuffled order is kept
// in the generated machine code.
std::vector<CompiledBinary> compileBinarySet(const llvm::Module &module,
                                             const CompileOptions &opts,
                                             std::string side,
                                             std::string *error = nullptr);

} // namespace mlir_mr
