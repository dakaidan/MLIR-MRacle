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

// Concise per-binary breakdown appended to run_info.json
struct BinaryOutcomeSummary {
    BinaryIdentity identity;
    ObservedOutcomeSet observed;
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
    std::vector<BinaryOutcomeSummary> binaryOutcomes;

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
