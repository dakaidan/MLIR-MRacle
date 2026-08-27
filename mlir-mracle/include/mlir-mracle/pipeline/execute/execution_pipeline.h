#pragma once

#include "mlir-mracle/execution/execution.h"
#include "mlir-mracle/pipeline/pipeline.h"

namespace mlir_mracle {

struct ExecutionPipelineResult {
    std::vector<ExecutionRunResult> runs;
    std::string campaignDir;
};

// single run of the execution pipeline for a single input file
ExecutionRunResult executeSingle(const std::string &inputFile, int seed,
                                 int runIdx, int reps);

// core pipeline function for --mode=execution
// calls executeSingle for each input file and each run, and collects the results
ExecutionPipelineResult runExecutionPipeline(const PipelineOptions &opts);

} // namespace mlir_mracle
