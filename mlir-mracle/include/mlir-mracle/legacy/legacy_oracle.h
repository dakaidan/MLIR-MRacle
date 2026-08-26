#pragma once

#include "mlir-mracle/context/context.h"

#include <cstdint>
#include <vector>
#include <cmath>

namespace mlir_mracle {

// Sorted difference lhs - rhs of joint outcome vectors.
std::vector<JointOutcome> outcomeSetDifference(
    const std::vector<JointOutcome> &lhs,
    const std::vector<JointOutcome> &rhs);

// Compares the observed outcome sets of the source and transformed programs.
// At 1 thread both sets must hold exactly one, identical outcome; any
// multi-valued set or mismatch is a failure. At higher thread counts the
// comparison is one-directional: the source distribution is the reference
// and only the transformed counts are judged. Every outcome in the union of
// the two sets is checked against the count the source rate predicts for the
// transformed batch; outcomes absent from the source are anchored on one
// expected chance occurrence. An outcome the source produces below
// thresholdPct percent of its runs that appears at or above that rate in the
// transformed batch fails only when its count also exceeds the Poisson upper
// bound of the expected count (p < 1e-6). Other transformed counts outside
// the two-sided Poisson range of the expected count (p < 1e-6) warn; the
// rest are ok.
OutcomeSetResult compareOutcomeSets(const ObservedOutcomeSet &source,
                                    const ObservedOutcomeSet &transformed,
                                    int numThreads, int thresholdPct);

// Subset-direction comparison: the transformed outcome set must be a subset
// of the source outcome set, so only outcomes the transformed side adds are
// judged. Missing source outcomes are allowed. A transformed-only outcome
// whose count is beyond the Poisson bound of a single chance occurrence and
// that appears at or above thresholdPct percent of transformed runs fails;
// one beyond the bound but below the threshold warns. The 1-thread
// determinism check is shared with compareOutcomeSets.
OutcomeSetResult compareOutcomeSetsSubset(
    const ObservedOutcomeSet &source, const ObservedOutcomeSet &transformed,
    int numThreads, int thresholdPct);

// Superset-direction comparison: the transformed outcome set must be a
// superset of the source outcome set, so only outcomes the transformed side
// drops are judged. Extra transformed outcomes are allowed. A source outcome
// produced at or above thresholdPct percent of source runs that is absent
// from the transformed set fails when the absence is also
// Poisson-significant; a rarer source outcome whose absence is statistically
// significant warns. The 1-thread determinism check is shared with
// compareOutcomeSets.
OutcomeSetResult compareOutcomeSetsSuperset(
    const ObservedOutcomeSet &source, const ObservedOutcomeSet &transformed,
    int numThreads, int thresholdPct);

} // namespace mlir_mracle
