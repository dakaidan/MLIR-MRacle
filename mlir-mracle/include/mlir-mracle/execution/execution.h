#pragma once

#include "mlir-mracle/agitation/agitation.h"
#include "mlir-mracle/core/types.h"

#include "llvm/IR/Module.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace mlir_mracle {

// Runs fn numRuns times from a single caller thread at the given OpenMP team
// size and returns the sorted unique joint outcomes observed, along with the
// number of times each one occurred.
ObservedOutcomeSet collectOutcomeSet(
    const std::function<std::vector<int64_t>()> &fn, int numRuns,
    int numThreads);

// Sorted merge of two observed outcome sets; counts and totalRuns are summed.
ObservedOutcomeSet mergeOutcomeSets(const ObservedOutcomeSet &a,
                                    const ObservedOutcomeSet &b);

// Pins the OpenMP runtime's process-wide settings before any parallel region
// initialises it: dynamic adjustment is off so omp_set_num_threads stays
// authoritative, and the default team size is fixed. Values are forced so a
// pre-existing environment cannot change them; idempotent.
void applyProcessSettings();

// Spreads totalRuns evenly across configs; the first totalRuns %
// configs.size() configs receive one extra run each.
void distributeRuns(std::vector<AgitationConfig> &configs, int totalRuns);

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
};

// A compiled, in-memory binary produced from a source module: the callable
// runs the module's parallel main and returns its scalar results.
struct CompiledBinary {
    std::string side; // "source" | "transformed"
    int compileIndex = 0;
    int jitOptLevel = -1;
    std::function<std::vector<int64_t>()> fn;
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

// Final triage once the plain rerun loop has exhausted its max-runs budget
// with rare states still unresolved: compiles one TSan-instrumented binary
// per side (TSan perturbs memory-access scheduling, so unresolved states
// either surface under instrumentation or are confirmed absent) and merges
// its observed outcomes into exec's aggregate totals and per-binary lists.
// extraRuns are spread across fresh agitation configs drawn from triageSeed.
// Returns false and sets error when the TSan runtime is unavailable; the
// caller then keeps the plain post-rerun verdict.
bool runTsanTriage(const llvm::Module &sourceModule,
                   const llvm::Module &transformedModule,
                   ExecutionResult &exec, int extraRuns, uint32_t triageSeed,
                   int configCount = 5, std::string *error = nullptr);

// Pointers to every compiled binary (source then transformed), in binary
// index order.
std::vector<const CompiledBinary *>
collectAllBinaries(const ExecutionResult &exec);

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
