#pragma once

#include "mlir-mracle/agitation/agitation.h"
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

// Pins the OpenMP runtime's process-wide settings
void applyProcessSettings();

// Spreads totalRuns evenly across configs
void distributeRuns(std::vector<AgitationConfig> &configs, int totalRuns);

struct ExecutionOptions {
    uint32_t seed = 42;
    int runsPerBinary = 5000;
    int singleThreadRuns = 32;
    // Number of distinct random OpenMP configs per binary.
    int configCount = 5;
};

// A compiled, in-memory binary and its accumulated outcomes
struct CompiledBinary {
    BinaryIdentity identity;
    std::function<std::vector<int64_t>()> fn;
    ObservedOutcomeSet singleThread;
    ObservedOutcomeSet threadedTotal;
};

struct ExecutionResult {
    std::vector<CompiledBinary> sourceBinaries;
    std::vector<CompiledBinary> transformedBinaries;

    ObservedOutcomeSet sourceTotal;
    ObservedOutcomeSet transformedTotal;
    ObservedOutcomeSet total;
    
    ObservedOutcomeSet sourceSingleThreadTotal;
    ObservedOutcomeSet transformedSingleThreadTotal;

    std::string error;
};

// Main orchestration of the execution of the source and transformed binaries
// Includes the agitation sweep, collection of outcomes, and merging of the results into the union outcome sets
ExecutionResult runExecutionHarness(const llvm::Module &sourceModule,
                                    const llvm::Module &transformedModule,
                                    const ExecutionOptions &opts);

// Re-runs every compiled binary (source and transformed) when results are inconclusive
// Used to make conclusions on rare states if possible, or to reach the --max-runs cap if not
void rerunAllBinaries(ExecutionResult &exec, int extraRuns,
                      uint32_t roundSeed, int configCount = 5);

// Final TSan triage once the plain rerun loop has exhausted its max-runs budget
bool runTsanTriage(const llvm::Module &sourceModule,
                   const llvm::Module &transformedModule,
                   ExecutionResult &exec, int extraRuns, uint32_t triageSeed,
                   int configCount = 5, std::string *error = nullptr);

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
