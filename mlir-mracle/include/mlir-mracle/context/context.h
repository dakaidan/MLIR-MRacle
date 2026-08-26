#pragma once

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/IR/LLVMContext.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace mlir_mracle {

// Direction of the outcome-set relation the comparison enforces. Equality
// judges the whole union of both outcome sets; Subset only judges outcomes
// the transformed side adds; Superset only judges outcomes the transformed
// side drops.
enum class OutcomeRelation { Equality, Subset, Superset };

std::string outcomeRelationToString(OutcomeRelation relation);
bool outcomeRelationFromString(const std::string &s,
                               OutcomeRelation &relation);

// A single metamorphic transformation applied during a run.
// Used for logging and reporting in RunInfo.
struct AppliedTransformation {
    std::string name;
    std::string targetFunction;
};

// Severity of a structured verdict issue: FAIL drives the overall ERROR
// status, WARN drives WARN; OK runs carry no issues.
enum class IssueSeverity { Fail, Warn };

std::string issueSeverityToString(IssueSeverity severity);

// one structured deviation reported by the oracle, replacing the free-form
// message: reason is the deviation category (e.g. "missing outcome"),
// outcome is the affected joint outcome/value when one exists, and severity
// classifies it
struct VerdictIssue {
    IssueSeverity severity = IssueSeverity::Warn;
    std::string outcome; // e.g. "[2]" or "var0=3"; empty when not applicable
    std::string reason;
};

// Result of a comparison between two outcome sets.
struct CompareResult {
    bool ok = true;
    bool warn = false;
    std::string message; // joined reasons, kept for legacy/internal tests
    std::vector<VerdictIssue> issues; // structured deviations, FAIL first

    CompareResult() = default;
    CompareResult(bool ok, bool warn, std::string message)
        : ok(ok), warn(warn), message(std::move(message)) {}
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
    std::vector<VerdictIssue> issues; // structured deviations, FAIL first
    int originalRuns = 0;      // executions of the source program in this group
    int transformedRuns = 0;   // executions of the transformed program in this group
    OutcomeSetResult outcomeSet;
};

// outcomes observed for a single compiled binary during an agitation sweep;
// the concise per-binary breakdown appended below the union outcome sets in
// the default run_info.json
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
    std::vector<ThreadGroupResult> threadResults;
    bool transformApplied = false;
    // union outcome sets aggregated over the whole agitation sweep (thread
    // counts, opt levels, code layout); filled by the default pipeline
    // instead of a fake thread group
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
    // structured deviations behind the status; the default pipeline copies
    // the oracle's issues here, legacy runs fall back to thread results
    std::vector<VerdictIssue> issues;
    // artifacts captured during the run; saved under results/<status>/ for
    // every run
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
              int maxApply = 1, std::string model = "");
};

} // namespace mlir_mracle
