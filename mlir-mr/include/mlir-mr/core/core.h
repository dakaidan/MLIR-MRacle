#pragma once

#include "mlir-mr/context/context.h"

#include <string>
#include <vector>

namespace mlir_mr {

// config for the runPipeline function
struct PipelineOptions {
    std::string inputFile;
    std::string multiFolder;
    int seed = -1;          // -1 = random per run, >=0 = fixed
    int runNumber = 0;
    int numRuns = 1;
    std::string transform;  // comma-separated list of transforms to try, empty = any
    int maxApply = 1;       // limit of transforms to apply per run
    bool printMLIR = false;
    bool log = false;       // output transformed MLIR, lowered MLIR, and JIT LLVM IR to logs
    bool verbose = false;   // include full outcome sets in warn/fail details
    int tsanPercent = 100;  // 0-100; share of compilations instrumented with TSan
    std::string campaignDir; // resume/continue an existing campaign log folder
};

struct PipelineResult {
    std::vector<RunInfo> runs;
};

PipelineResult runPipeline(const PipelineOptions &opts);

} // namespace mlir_mr
