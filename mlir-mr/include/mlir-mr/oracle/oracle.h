#pragma once

#include "mlir-mr/context/context.h"

#include <cstdint>
#include <functional>
#include <vector>
#include <cmath>

namespace mlir_mr {

// OpenMP team-size sweep per run; the 2-thread primary test runs first
inline constexpr int kThreadCounts[] = {2, 1, 4, 8};

// Runs fn numRuns times from a single caller thread at the given OpenMP team
// size and returns the sorted unique joint outcomes observed, along with the
// number of times each one occurred.
ObservedOutcomeSet collectOutcomeSet(
    const std::function<std::vector<int64_t>()> &fn, int numRuns,
    int numThreads);

// Sorted merge of two observed outcome sets; counts and totalRuns are summed.
ObservedOutcomeSet mergeOutcomeSets(const ObservedOutcomeSet &a,
                                    const ObservedOutcomeSet &b);

// Sorted difference lhs - rhs of joint outcome vectors.
std::vector<JointOutcome> outcomeSetDifference(
    const std::vector<JointOutcome> &lhs,
    const std::vector<JointOutcome> &rhs);

// Compares the observed outcome sets of the source and transformed programs.
// At 1 thread both sets must hold exactly one, identical outcome; any
// multi-valued set or mismatch is a failure. At higher thread counts,
// transformed-only outcomes that are rare in the transformed batch (below
// thresholdPct percent of its runs) warn; the rest fail. Outcomes unique to
// the source warn.
OutcomeSetResult compareOutcomeSets(const ObservedOutcomeSet &source,
                                    const ObservedOutcomeSet &transformed,
                                    int numThreads, int thresholdPct);

} // namespace mlir_mr
