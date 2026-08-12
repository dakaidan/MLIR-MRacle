#pragma once

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/IR/LLVMContext.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace mlir_mr {

// A single metamorphic transformation applied during a run.
// Used for logging and reporting in RunInfo.
struct AppliedTransformation {
    std::string name;
    std::string targetFunction;
};

// Result of a comparison between two outcome sets.
struct CompareResult {
    bool ok = true;
    bool warn = false;
    std::string message;
};

// Result of Fisher's exact test (Freeman-Halton extension) for one output
// variable. pValue is the two-sided Monte Carlo estimate; pLow/pHigh are
// the 99% confidence interval used for adaptive stopping. categories and the
// count vectors are aligned.
struct FisherResult {
    double pValue = 1.0;
    double pLow = 1.0;
    double pHigh = 1.0;
    int simulations = 0;
    std::vector<int64_t> categories;
    std::vector<int> sourceCounts;
    std::vector<int> transformedCounts;
    std::vector<int64_t> novelOutcomes;       // only in transformed
    std::vector<int64_t> disappearedOutcomes; // only in source
};

// classification of a p-value against the fail/warn thresholds
enum class FisherVerdict { Pass, Warn, Fail };

// result of a sequential batch comparison between two programs
struct SequentialResult {
    FisherVerdict verdict = FisherVerdict::Pass;
    int sourceRuns = 0;
    int transformedRuns = 0;
    std::vector<FisherResult> variables; // per-output, from the final batch
};

// Comparison result for a single thread-count group of a run.
struct ThreadGroupResult {
    int numThreads;
    CompareResult comparison;
    int originalRuns = 0;    // executions of the source program in this group
    int transformedRuns = 0; // executions of the transformed program in this group
    std::vector<FisherResult> variables; // per-output Fisher results, empty for t==1
};

// Main data object about a single run of the metamorphic testing pipeline.
struct RunInfo {
    int runNumber = 0;
    int seed = 42;
    std::string file;
    std::vector<std::string> requestedTransforms;
    std::vector<AppliedTransformation> appliedTransforms;
    std::vector<ThreadGroupResult> threadResults;
    bool transformApplied = false;
    std::string error;
    std::string warn;
    std::string mlirOutput;
};

struct MLIRSetup {
    mlir::MLIRContext mlirContext;
    llvm::LLVMContext llvmContext;
    mlir::PassManager pm;
    RunInfo runInfo;

    MLIRSetup(int seed = 42, int runNumber = 0, std::string transform = "",
              int maxApply = 1);
};

} // namespace mlir_mr
