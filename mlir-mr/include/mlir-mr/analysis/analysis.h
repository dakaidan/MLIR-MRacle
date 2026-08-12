#pragma once

#include "mlir-mr/context/context.h"
#include "mlir-mr/outcome/outcome.h"

#include <map>
#include <string>
#include <vector>

namespace mlir_mr {

// comparison strategy between original and transformed outcome sets
// arch: hard-coded to Equality for now; Subset/Superset reserved for future
// metamorphic relations that tolerate novel or disappeared outcomes
enum class Relation {
    Equality,  // transformed must equal original
    Subset,    // transformed must be a subset of original
    Superset,  // transformed must be a superset of original
};

// threshold for a novel outcome to be considered a genuine behavioural change
inline constexpr double kNovelOutcomeFrequency = 0.05;

// data object describing comparison issues between two outcome sets
struct VariableIssue {
    std::string label;
    std::string originalSet;
    std::string transformedSet;
    std::vector<std::string> notes;  // novel/disappeared detail lines
    bool disjoint = false;
    bool hardFail = false;
};

// Compares the outcome counts of two modules and returns a summary of the comparison
//
// - If a novel outcome appears more than kNovelOutcomeFrequency -> hard fail
// - If the outcome sets are disjoint -> hard fail
// - If a novel outcome appears but below the threshold -> warn
// - If an outcome disappears more than kNovelOutcomeFrequency -> hard fail
// - If the outcome sets are identical -> ok
//
// The relation controls which of these checks apply: equality forbids both
// novel and disappeared outcomes, subset only novel ones, superset only
// disappeared ones.
CompareResult compareOutcomes(const OutcomeCounts &originalCounts,
                              const OutcomeCounts &transformedCounts,
                              int originalRuns, int transformedRuns,
                              bool verbose, Relation relation);

// renders an outcome set as {v1, v2, ...}
std::string formatOutcomeSet(const std::map<int64_t, int> &counts);

// verbosity-aware rendering of the issues gathered during comparison
CompareResult renderComparison(int originalRuns, int transformedRuns,
                               bool verbose,
                               const std::vector<VariableIssue> &issues);

} // namespace mlir_mr
