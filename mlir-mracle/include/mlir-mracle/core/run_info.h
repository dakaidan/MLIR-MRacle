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

// Verdict of the distributional oracle and the protocol that produced it
struct DistributionSummary {
    std::string verdict;    // reported verdict: "OK" | "BAD" | "WARN"
    std::string tvdVerdict; // distributional component alone: "OK" | "BAD" | "INCONCLUSIVE"
    double tvd = 0.0;
    double ciLow = 0.0;
    double ciHigh = 0.0;
    double tvdThreshold = 0.05;
    double confidenceLevel = 0.95;
    int bootstrapReplicates = 0;
    int64_t sourceCampaigns = 0;
    int64_t transformedCampaigns = 0;
    int64_t excludedCampaigns = 0;
    int64_t sourceRuns = 0;
    int64_t transformedRuns = 0;
    int64_t sourceSupport = 0;
    int64_t transformedSupport = 0;
    double sourceEffCategories = 0.0;
    double transformedEffCategories = 0.0;
    double sourceUnseenMass = 0.0; // estimated mass on outcomes never observed
    double transformedUnseenMass = 0.0;

    // measurement protocol, recorded so a run can be reproduced
    int executionsPerCampaign = 0;
    int teamSize = 0;
    int jitOptLevel = -1;
};

// An outcome only one variant produced, with the evidence gathered for it
struct SupportViolationSummary {
    std::string outcome;      // rendered joint outcome, e.g. "[1, 2]"
    std::string side;         // offending side: "source" | "transformed"
    int64_t count = 0;
    int64_t runs = 0;
    double rate = 0.0;
    double bandRate = 0.0;    // fluke band the outcome was judged against
    bool significant = false;
    bool establishable = true;
    std::string verdict;      // "FAIL" | "CANDIDATE" | "OK"
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

    // distributional verdict of the default pipeline; verdict is empty for the
    // pipelines that do not compare distributions
    DistributionSummary distribution;
    std::vector<SupportViolationSummary> supportViolations;
    
    // artifacts captured during the run;
    std::string sourceMLIR;
    std::string transformedMLIR;
    std::string loweredMLIR;
    std::string jitLLVM;
    std::string sourceJitLLVM;
    std::string bitcode;
};

} // namespace mlir_mracle
