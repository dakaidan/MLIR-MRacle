#pragma once

#include "mlir-mr/context/context.h"

#include <map>
#include <string>
#include <vector>
#include <cmath>

namespace mlir_mr {

// outcome frequencies map for each output variable of a module
using OutcomeCounts = std::vector<std::map<int64_t, int>>;

// data object describing comparison issues between two outcome sets
struct VariableIssue {
    std::string label;
    std::string originalSet;
    std::string transformedSet;
    std::vector<std::string> notes;  // novel/disappeared detail lines
    bool disjoint = false;
    bool hardFail = false;
};

struct CompareResult {
    bool ok = true;
    bool warn = false;
    std::string message;
};

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
};

struct PipelineResult {
    std::vector<RunInfo> runs;
};

PipelineResult runPipeline(const PipelineOptions &opts);

} // namespace mlir_mr
