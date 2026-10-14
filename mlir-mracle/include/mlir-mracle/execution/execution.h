#pragma once

#include "mlir-mracle/core/types.h"

#include "llvm/IR/Module.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace mlir_mracle {

// Runs fn() numRuns times and collects the outcome set
ObservedOutcomeSet collectOutcomeSet(
    const std::function<std::vector<int64_t>()> &fn, int numRuns,
    int numThreads); // sizes the OpenMP team to numThreads before running fn() in parallel; see implementation

// Sorted merge of two observed outcome sets; counts and totalRuns are summed.
ObservedOutcomeSet mergeOutcomeSets(const ObservedOutcomeSet &a,
                                    const ObservedOutcomeSet &b);

// Pins the OpenMP runtime's process-wide settings for a run
void applyProcessSettings(int teamSize = 2);

// The measurement protocol of one campaign: a fixed compilation configuration
// and a single runtime configuration, so campaigns differ only by the
// compilation instance and the executions themselves.
struct CampaignOptions {
    int executionsPerCampaign = 2000;
    int teamSize = 4;              // OpenMP team size for every execution
    int jitOptLevel = -1;          // JIT default CodeGen level
    int determinismProbeRuns = 32; // 1-thread probe executions, 0 disables the probe
};

// A compiled, in-memory binary and its accumulated outcomes
struct CompiledBinary {
    BinaryIdentity identity;
    std::function<std::vector<int64_t>()> fn;
    ObservedOutcomeSet threadedTotal;
};

struct ExecutionResult {
    std::vector<CompiledBinary> sourceBinaries;
    std::vector<CompiledBinary> transformedBinaries;

    ObservedOutcomeSet sourceTotal;
    ObservedOutcomeSet transformedTotal;
    ObservedOutcomeSet total;

    // 1-thread determinism probe of each side, collected once per run
    ObservedOutcomeSet sourceSingleThreadTotal;
    ObservedOutcomeSet transformedSingleThreadTotal;

    std::string error;
};

// Compiles and runs one fresh campaign per side, appending them to exec. Sides
// alternate, and the first campaign also carries the 1-thread probe. Returns
// false with error set when either side fails to compile.
bool runCampaignPair(ExecutionResult &exec, const llvm::Module &sourceModule,
                     const llvm::Module &transformedModule, int campaignIndex,
                     uint32_t campaignSeed, const CampaignOptions &opts,
                     std::string &error);


// outcomes observed at one OpenMP team size in execution mode
struct ExecutionThreadResult {
    int numThreads = 0;
    int runs = 0;
    std::vector<JointOutcome> outcomes; // sorted unique joint outcomes
    std::vector<int64_t> counts;        // occurrences per outcome, parallel
};

// one execution-mode run of a single source file; kept separate from RunInfo
// because execution mode has no transforms, comparison, or status classes
struct ExecutionRunResult {
    int runNumber = 0;
    std::string file;
    int seed = 42;
    std::vector<ExecutionThreadResult> threadResults;
    // LLVM IR of the lowered source program; saved as the run's .ll artifact
    std::string llvmIR;
    std::string error;
};

} // namespace mlir_mracle
