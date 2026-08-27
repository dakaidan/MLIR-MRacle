#pragma once

#include "mlir-mracle/core/types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir_mracle {

// Info object for a single metamorphic transformation that has been applied during a run.
struct AppliedTransformation {
    std::string name;
    std::string targetFunction;
};

// Legacy comparison result for a single thread-count group of a run.
struct ThreadGroupResult {
    int numThreads;
    std::string status = "OK"; // "OK" | "WARN" | "ERROR"
    std::string message;       // short category, not serialized to JSON
    std::vector<VerdictIssue> issues; // vector of issues found in the comparison
    int originalRuns = 0;      // executions of the source program in this group
    int transformedRuns = 0;   // executions of the transformed program in this group
    OutcomeSetResult outcomeSet;
};

// Outcomes observed for a single compiled binary during an agitation sweep
struct BinaryOutcomeResult {
    std::string side;      // "source" | "transformed"
    int compileIndex = 0;  // index within the side's binary set
    int jitOptLevel = -1;  // CodeGen opt level used for this binary
    int runs = 0;          // executions represented by outcomes/counts
    std::vector<JointOutcome> outcomes;
    std::vector<int64_t> counts;
};

// Main data object about a single run of the metamorphic testing pipeline.
struct RunInfo {
    int runNumber = 0;
    int seed = 42;
    std::string file;

    std::vector<std::string> requestedTransforms;
    std::vector<AppliedTransformation> appliedTransforms;
    bool transformApplied = false;

    // Legacy results
    std::vector<ThreadGroupResult> threadResults;

    int64_t sourceRuns = 0;
    int64_t transformedRuns = 0;

    std::vector<JointOutcome> sourceOutcomes;
    std::vector<int64_t> sourceCounts;
    std::vector<JointOutcome> transformedOutcomes;
    std::vector<int64_t> transformedCounts;
    std::vector<BinaryOutcomeResult> binaryOutcomes;

    OutcomeRelation relation = OutcomeRelation::Equality;
    std::string error;
    std::string warn;
    std::vector<VerdictIssue> issues;
    
    // artifacts captured during the run;
    std::string sourceMLIR;
    std::string transformedMLIR;
    std::string loweredMLIR;
    std::string jitLLVM;
    std::string sourceJitLLVM;
    std::string bitcode;
};

} // namespace mlir_mracle
