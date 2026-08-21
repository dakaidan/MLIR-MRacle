#pragma once

#include "mlir-mracle/agitation/agitation.h"
#include "mlir-mracle/context/context.h"
#include "mlir-mracle/oracle/oracle.h"

#include "llvm/IR/Module.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir_mracle {

struct BinaryConfigResult {
    int configIndex = 0;
    AgitationConfig config;
    ObservedOutcomeSet outcomes; // separate per config
};

struct BinaryExecutionResult {
    int binaryIndex = 0;
    std::string side;
    ObservedOutcomeSet singleThread;  // determinism check, separate
    ObservedOutcomeSet threadedTotal; // merged across configs
    std::vector<BinaryConfigResult> perConfig;
};

struct ExecutionOptions {
    uint32_t seed = 42;
    int runsPerBinary = 5000;
    int singleThreadRuns = 32;
    // Number of distinct random OpenMP configs per binary.
    int configCount = 5;
    // OMP_DYNAMIC is read only at OpenMP runtime initialisation, so it is
    // pinned once process-wide (see applyProcessSettings); per-config
    // variation uses omp_set_num_threads only.
    CompileOptions compile;
};

struct ExecutionResult {
    std::vector<CompiledBinary> sourceBinaries;
    std::vector<CompiledBinary> transformedBinaries;
    std::vector<BinaryExecutionResult> binaryResults;

    // the agitation configs used for the initial batch; replay rounds
    // generate a fresh team-size mix per round, so this set is diagnostic
    // only
    std::vector<AgitationConfig> configs;

    ObservedOutcomeSet sourceTotal;
    ObservedOutcomeSet transformedTotal;
    ObservedOutcomeSet total;
    ObservedOutcomeSet sourceSingleThreadTotal;
    ObservedOutcomeSet transformedSingleThreadTotal;

    std::string error;
};

// Compiles sourceModule and transformedModule into 5 in-memory binaries each
// (varying jitOptLevel and basic-block layout) and runs every binary under
// the agitation configs. Returns the per-config, per-binary and aggregated
// outcome sets; the binaries stay alive for the lifetime of the result.
ExecutionResult runExecutionHarness(const llvm::Module &sourceModule,
                                    const llvm::Module &transformedModule,
                                    const ExecutionOptions &opts);

// Re-runs every compiled binary (source and transformed) for extraRuns total
// additional executions, going through the agitation process again: a fresh
// team-size mix is generated from roundSeed and the extra runs are spread
// across it. The per-binary totals are merged, then the aggregated outcome
// sets are recomputed. The binaries themselves are not recompiled (code
// generation is deterministic for a fixed module, and the initial sweep
// already samples the compile-variance axis). The single-thread determinism
// probes are not replayed; they are a fixed check that the replay round never
// invalidates.
void rerunAllBinaries(ExecutionResult &exec, int extraRuns,
                      uint32_t roundSeed, int configCount = 5);

} // namespace mlir_mracle
