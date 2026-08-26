#include "mlir-mracle/legacy/legacy_oracle.h"
#include "mlir-mracle/oracle/statistics.h"

#include <algorithm>
#include <iterator>
#include <map>
#include <vector>
#include <cmath>

namespace mlir_mracle {

namespace {

using oracle_detail::arityCompatible;
using oracle_detail::kStrongPValue;
using oracle_detail::outcomeCount;
using oracle_detail::poissonSurvival;

// expected count of a novel outcome (absent from the source set) under the
// null hypothesis: a single chance occurrence is treated as plausible noise
constexpr double kNovelExpected = 1.0;

// 1-thread groups are a determinism check for every relation: both sides
// must produce exactly the same single outcome
CompareResult singleThreadVerdict(const ObservedOutcomeSet &source,
                                  const ObservedOutcomeSet &transformed,
                                  const std::vector<JointOutcome> &sourceOnly,
                                  const std::vector<JointOutcome> &transformedOnly) {
    bool singleMatch = source.outcomes.size() == 1 &&
                       transformed.outcomes.size() == 1 &&
                       source.outcomes.front() == transformed.outcomes.front();
    if (singleMatch)
        return {true, false, ""};
    if (sourceOnly.empty() && transformedOnly.empty())
        return {false, false, "non-deterministic with 1 thread"};
    return {false, false, "result mismatch"};
}

// judge a single outcome against the count the source rate predicts for the
// transformed batch; returns {fail, warn}. An outcome absent from the source
// is anchored on one expected chance occurrence. An outcome the source
// produces below thresholdPct percent of its runs that the transformed batch
// produces at or above that rate fails; other counts outside the two-sided
// Poisson range of the expected count (p < 1e-6) warn.
std::pair<bool, bool> judgeOutcomeRate(const ObservedOutcomeSet &source,
                                       const ObservedOutcomeSet &transformed,
                                       const JointOutcome &o,
                                       int thresholdPct) {
    int64_t srcCount = outcomeCount(source, o);
    int64_t trCount = outcomeCount(transformed, o);
    const int64_t srcTotal = std::max<int64_t>(1, source.totalRuns);
    const int64_t trTotal = std::max<int64_t>(1, transformed.totalRuns);
    double expected = (srcCount == 0)
                          ? kNovelExpected
                          : (static_cast<double>(srcCount) / srcTotal) *
                                trTotal;
    double trPct = 100.0 * trCount / trTotal;
    if (100.0 * srcCount / srcTotal < thresholdPct && trPct >= thresholdPct)
        return {true, false};
    double pHigh = poissonSurvival(trCount - 1, expected);
    double pLow = 1.0 - poissonSurvival(trCount, expected);
    if (pHigh < kStrongPValue || pLow < kStrongPValue)
        return {false, true};
    return {false, false};
}

// judge a source outcome that the transformed set does not contain: a common
// source outcome (>= thresholdPct of source runs) fails outright, a rarer one
// warns when its absence is statistically significant relative to the
// source-predicted count
std::pair<bool, bool> judgeMissingOutcome(const ObservedOutcomeSet &source,
                                          const ObservedOutcomeSet &transformed,
                                          const JointOutcome &o,
                                          int thresholdPct) {
    int64_t srcCount = outcomeCount(source, o);
    const int64_t srcTotal = std::max<int64_t>(1, source.totalRuns);
    const int64_t trTotal = std::max<int64_t>(1, transformed.totalRuns);
    if (100.0 * srcCount / srcTotal >= thresholdPct)
        return {true, false};
    double expected =
        (static_cast<double>(srcCount) / srcTotal) * trTotal;
    // absence is P(X = 0) = e^-expected under the source-predicted rate
    if (std::exp(-expected) < kStrongPValue)
        return {false, true};
    return {false, false};
}

CompareResult verdictFromFlags(bool anyFail, bool anyWarn) {
    if (anyFail)
        return {false, false, "behavioural change detected"};
    if (anyWarn)
        return {true, true, "outcome rate shift detected"};
    return {true, false, ""};
}

} // namespace

std::vector<JointOutcome> outcomeSetDifference(
    const std::vector<JointOutcome> &lhs,
    const std::vector<JointOutcome> &rhs) {
    std::vector<JointOutcome> out;
    std::set_difference(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
                        std::back_inserter(out));
    return out;
}

// set difference lhs - rhs, shared by the comparison functions below
static void setDifferenceInto(const ObservedOutcomeSet &lhs,
                              const ObservedOutcomeSet &rhs,
                              std::vector<JointOutcome> &out) {
    std::set_difference(lhs.outcomes.begin(), lhs.outcomes.end(),
                        rhs.outcomes.begin(), rhs.outcomes.end(),
                        std::back_inserter(out));
}

OutcomeSetResult compareOutcomeSets(const ObservedOutcomeSet &source,
                                    const ObservedOutcomeSet &transformed,
                                    int numThreads, int thresholdPct) {
    OutcomeSetResult result;
    result.source = source;
    result.transformed = transformed;

    if (!arityCompatible(source, transformed)) {
        result.compare = {false, false, "result arity mismatch"};
        return result;
    }

    std::vector<JointOutcome> sourceOnly;
    std::vector<JointOutcome> transformedOnly;
    setDifferenceInto(source, transformed, sourceOnly);
    setDifferenceInto(transformed, source, transformedOnly);

    if (numThreads == 1) {
        result.compare = singleThreadVerdict(source, transformed, sourceOnly,
                                             transformedOnly);
        return result;
    }

    // one-directional comparison: the source distribution is the reference,
    // so only the transformed counts are judged, never the source counts.
    // Every outcome in the union of the two sets is checked against the
    // count the source rate predicts for the transformed batch.
    bool anyFail = false;
    bool anyWarn = false;
    std::vector<JointOutcome> all;
    std::set_union(source.outcomes.begin(), source.outcomes.end(),
                   transformed.outcomes.begin(), transformed.outcomes.end(),
                   std::back_inserter(all));
    for (const auto &outcome : all) {
        auto [fail, warn] =
            judgeOutcomeRate(source, transformed, outcome, thresholdPct);
        anyFail |= fail;
        anyWarn |= warn;
    }
    result.compare = verdictFromFlags(anyFail, anyWarn);
    return result;
}

OutcomeSetResult compareOutcomeSetsSubset(
    const ObservedOutcomeSet &source, const ObservedOutcomeSet &transformed,
    int numThreads, int thresholdPct) {
    OutcomeSetResult result;
    result.source = source;
    result.transformed = transformed;

    if (!arityCompatible(source, transformed)) {
        result.compare = {false, false, "result arity mismatch"};
        return result;
    }

    std::vector<JointOutcome> sourceOnly;
    std::vector<JointOutcome> transformedOnly;
    setDifferenceInto(source, transformed, sourceOnly);
    setDifferenceInto(transformed, source, transformedOnly);

    if (numThreads == 1) {
        result.compare = singleThreadVerdict(source, transformed, sourceOnly,
                                             transformedOnly);
        return result;
    }

    // subset direction: the transformed set must not add outcomes, so only
    // transformed-only outcomes are judged; missing source outcomes are
    // allowed. A transformed-only outcome is a novel outcome anchored on one
    // expected chance occurrence.
    bool anyFail = false;
    bool anyWarn = false;
    for (const auto &outcome : transformedOnly) {
        auto [fail, warn] =
            judgeOutcomeRate(source, transformed, outcome, thresholdPct);
        anyFail |= fail;
        anyWarn |= warn;
    }
    result.compare = verdictFromFlags(anyFail, anyWarn);
    return result;
}

OutcomeSetResult compareOutcomeSetsSuperset(
    const ObservedOutcomeSet &source, const ObservedOutcomeSet &transformed,
    int numThreads, int thresholdPct) {
    OutcomeSetResult result;
    result.source = source;
    result.transformed = transformed;

    if (!arityCompatible(source, transformed)) {
        result.compare = {false, false, "result arity mismatch"};
        return result;
    }

    std::vector<JointOutcome> sourceOnly;
    std::vector<JointOutcome> transformedOnly;
    setDifferenceInto(source, transformed, sourceOnly);
    setDifferenceInto(transformed, source, transformedOnly);

    if (numThreads == 1) {
        result.compare = singleThreadVerdict(source, transformed, sourceOnly,
                                             transformedOnly);
        return result;
    }

    // superset direction: the transformed set must keep every source outcome,
    // so only source-only outcomes are judged; extra transformed outcomes are
    // allowed.
    bool anyFail = false;
    bool anyWarn = false;
    for (const auto &outcome : sourceOnly) {
        auto [fail, warn] =
            judgeMissingOutcome(source, transformed, outcome, thresholdPct);
        anyFail |= fail;
        anyWarn |= warn;
    }
    result.compare = verdictFromFlags(anyFail, anyWarn);
    return result;
}

} // namespace mlir_mracle
