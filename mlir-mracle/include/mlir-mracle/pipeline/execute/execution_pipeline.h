#pragma once

#include "mlir-mracle/pipeline/pipeline.h"

namespace mlir_mracle {

struct ExecutionPipelineResult {
    std::vector<ExecutionRunResult> runs;
    std::string campaignDir;
};

// straight execution mode: parse, lower, JIT and run the source program at
// every team size without any transformation or comparison. Joint outcome
// frequencies are recorded per thread count. Every call compiles and runs
// fresh: execution mode is a probe of the run, not a comparison, so nothing
// is memoised or cached across runs.
ExecutionRunResult executeSingle(const std::string &inputFile, int seed,
                                 int runIdx, int reps);

// core pipeline function for --mode=execution: executes each input file
// as-is and
// records joint outcome frequencies per thread count. Each run is published
// to the campaign folder as it completes; there is no status classification.
ExecutionPipelineResult runExecutionPipeline(const PipelineOptions &opts);

} // namespace mlir_mracle
