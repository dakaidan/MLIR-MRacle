#pragma once

#include "mlir-mracle/core/run_info.h"
#include "mlir-mracle/core/types.h"

#include <string>
#include <vector>
#include <cmath>
#include <set>

namespace mlir_mracle {

// config for the runPipeline function
struct PipelineOptions {
    std::string inputFile;
    std::string multiFolder;
    int seed = -1;
    int runNumber = 0;          // first run index
    int numRuns = 1;            // pipeline repetitions, set by --iter
    int reps = 5000;            // executions per program per thread count, set by --reps
    std::string transform;      // comma-separated list of transforms to try, empty = any
    int maxApply = 1;           // limit of transforms to apply per run
    std::string model;          // memory model gating applicable transforms, empty = generic only
    std::string campaignDir;    // output folder; runs are added as they complete
    int retestReps = 5000;      // extra source runs when transformed finds a new outcome
    int maxSourceReps = 100000; // hard cap for total source runs across all binaries
    int thresholdPct = 5;       // fail/warn classifier
};

// result of a single run of the pipeline, including the verdict and all outcome sets
struct PipelineResult {
    std::vector<RunInfo> runs;
    std::string campaignDir;
};

// core pipeline function
// applies transforms, runs the harness, and returns the verdict and all outcome sets
PipelineResult runPipeline(const PipelineOptions &opts);

// single run of the pipeline
// is also called by runPipeline for each repetition
RunInfo runSingle(const std::string &inputFile, int seed, int runIdx,
                  const PipelineOptions &opts);

// releases process-lifetime JIT state (compiled source binaries) while the
// runtime is still healthy; must be called before exit
void shutdownPipeline();

} // namespace mlir_mracle
