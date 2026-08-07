#pragma once

#include "mlir-mr/context/context.h"

#include <string>
#include <vector>

namespace mlir_mr {

struct PipelineOptions {
    std::string inputFile;
    std::string multiFolder;
    int seed = -1;          // -1 = random per run, >=0 = fixed
    int runNumber = 0;
    int numRuns = 1;
    std::string transform;  // comma-separated list of transforms to try, empty = any
    int maxApply = 1;       // the limit on the number of times a transformation can be applied to a single function, 1 is default
    bool printMLIR = false;
};

struct PipelineResult {
    std::vector<RunInfo> runs;
};

PipelineResult runPipeline(const PipelineOptions &opts);

} // namespace mlir_mr
