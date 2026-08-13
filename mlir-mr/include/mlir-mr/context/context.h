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

// one ordered tuple of outputs produced by a single execution
using JointOutcome = std::vector<int64_t>;

// sorted unique joint outcomes observed for one program over a batch;
// counts[i] is the number of times outcomes[i] was observed
struct ObservedOutcomeSet {
    std::vector<JointOutcome> outcomes;
    std::vector<int64_t> counts;
    size_t arity = 0;
    bool arityConsistent = true;
    int64_t totalRuns = 0; // executions represented by outcomes/counts
};

// outcome-set comparison between the source and transformed programs
struct OutcomeSetResult {
    ObservedOutcomeSet source;
    ObservedOutcomeSet transformed;
    CompareResult compare;
};

// Comparison result for a single thread-count group of a run.
struct ThreadGroupResult {
    int numThreads;
    std::string status = "OK"; // "OK" | "WARN" | "ERROR"
    std::string message;       // short category, not serialized to JSON
    int originalRuns = 0;      // executions of the source program in this group
    int transformedRuns = 0;   // executions of the transformed program in this group
    OutcomeSetResult outcomeSet;
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
    // artifacts captured during the run; saved under results/<status>/ for
    // warn/fail runs (and OK runs when --log is set)
    std::string sourceMLIR;
    std::string transformedMLIR;
    std::string loweredMLIR;
    std::string jitLLVM;
    std::string sourceJitLLVM;
    std::string bitcode;
};

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

// registers all dialects and LLVM IR translations shared by every pipeline
void initializeMLIRContext(mlir::MLIRContext &ctx);

struct MLIRSetup {
    mlir::MLIRContext mlirContext;
    llvm::LLVMContext llvmContext;
    mlir::PassManager pm;
    RunInfo runInfo;

    MLIRSetup(int seed = 42, int runNumber = 0, std::string transform = "",
              int maxApply = 1);
};

} // namespace mlir_mr
