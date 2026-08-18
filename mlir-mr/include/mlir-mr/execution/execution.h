#pragma once

#include "mlir-mr/agitation/agitation.h"
#include "mlir-mr/context/context.h"
#include "mlir-mr/oracle/oracle.h"

#include "llvm/IR/Module.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir_mr {

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
    // OMP_PROC_BIND and GOMP_CPU_AFFINITY are read only at OpenMP runtime
    // initialisation, so they are applied once process-wide with fixed values
    // (see applyProcessSettings); per-config variation uses the omp_set_*
    // runtime APIs only.
    CompileOptions compile;
};

struct ExecutionResult {
    std::vector<CompiledBinary> sourceBinaries;
    std::vector<CompiledBinary> transformedBinaries;
    std::vector<BinaryExecutionResult> binaryResults;

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

} // namespace mlir_mr
