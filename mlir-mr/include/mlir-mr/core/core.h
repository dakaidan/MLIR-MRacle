#pragma once

#include "mlir-mr/context/context.h"

#include <string>
#include <vector>
#include <cmath>

namespace mlir_mr {

// config for the runPipeline function
struct PipelineOptions {
    std::string inputFile;
    std::string multiFolder;
    int seed = -1;            // -1 = random per run, >=0 = fixed
    int runNumber = 0;        // first run index
    int numRuns = 1;          // pipeline repetitions, set by --iter
    bool straightMode = false; // --run: execute each file, no transforms
    bool emitMLIR = false;    // --emit-mlir: transform and emit MLIR only
    int reps = 5000;           // --reps: executions per program per thread count
    std::string transform;    // comma-separated list of transforms to try, empty = any
    int maxApply = 1;         // limit of transforms to apply per run
    int tsanPercent = 100;    // 0-100; % of compilations instrumented with TSan
    std::string campaignDir;  // output folder; runs are added as they complete
    int retestReps = 5000;    // extra source runs when transformed finds a new outcome
    int maxSourceReps = 100000; // hard cap for source runs per baseline
    int thresholdPct = 5; // transformed-only outcomes at/above this % fail, below warn
};

// schema version embedded in persistent baseline-cache keys and prefixes
inline constexpr int kResultSchemaVersion = 8;

struct PipelineResult {
    std::vector<RunInfo> runs;
    std::string campaignDir;
};

PipelineResult runPipeline(const PipelineOptions &opts);
PipelineResult runEmitPipeline(const PipelineOptions &opts);

struct ExecutionPipelineResult {
    std::vector<ExecutionRunResult> runs;
    std::string campaignDir;
};

ExecutionPipelineResult runExecutionPipeline(const PipelineOptions &opts);

// releases process-lifetime JIT state (compiled source binaries) while the
// runtime is still healthy; must be called before exit
void shutdownCore();

} // namespace mlir_mr
