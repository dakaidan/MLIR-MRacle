#include "mlir-mr/oracle/oracle.h"

#include <algorithm>
#include <iterator>
#include <map>
#include <omp.h>
#include <vector>
#include <cmath>

namespace mlir_mr {

namespace {

// p-value below which an outcome count is flagged as above the Poisson
// upper bound of the expected count
constexpr double kStrongPValue = 1e-6;

// expected count of a novel outcome (absent from the source set) under the
// null hypothesis: a single chance occurrence is treated as plausible noise
constexpr double kNovelExpected = 1.0;

// regularised lower incomplete gamma P(a,x) = gamma(a,x)/Gamma(a): series for
// x < a+1, continued fraction for the upper tail otherwise
double lowerIncompleteGamma(double a, double x) {
    constexpr double kEps = 1e-15;
    constexpr int kItMax = 200;
    constexpr double kFpMin = 1e-300;
    const double gln = std::lgamma(a);
    if (x < a + 1.0) {
        double ap = a;
        double sum = 1.0 / a;
        double del = sum;
        for (int n = 0; n < kItMax; ++n) {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if (std::fabs(del) < std::fabs(sum) * kEps)
                break;
        }
        return sum * std::exp(-x + a * std::log(x) - gln);
    }
    double b = x + 1.0 - a;
    double c = 1.0 / kFpMin;
    double d = 1.0 / b;
    double h = d;
    for (int i = 1; i <= kItMax; ++i) {
        double an = -static_cast<double>(i) * (static_cast<double>(i) - a);
        b += 2.0;
        d = an * d + b;
        if (std::fabs(d) < kFpMin)
            d = kFpMin;
        c = b + an / c;
        if (std::fabs(c) < kFpMin)
            c = kFpMin;
        d = 1.0 / d;
        double del = d * c;
        h *= del;
        if (std::fabs(del - 1.0) < kEps)
            break;
    }
    return 1.0 - std::exp(-x + a * std::log(x) - gln) * h;
}

// P(X > k) for X ~ Poisson(lambda) == scipy.stats.poisson.sf(k, lambda); uses
// P(Poisson(lambda) >= a) = P(a, lambda) for integer a
double poissonSurvival(int64_t k, double lambda) {
    if (k < 0)
        return 1.0;
    if (lambda <= 0.0)
        return 0.0;
    return lowerIncompleteGamma(static_cast<double>(k) + 1.0, lambda);
}

// number of runs of `set` that produced `o` (0 if absent)
int64_t outcomeCount(const ObservedOutcomeSet &set, const JointOutcome &o) {
    auto it = std::lower_bound(set.outcomes.begin(), set.outcomes.end(), o);
    if (it == set.outcomes.end() || *it != o)
        return 0;
    size_t idx = static_cast<size_t>(std::distance(set.outcomes.begin(), it));
    return idx < set.counts.size() ? set.counts[idx] : 0;
}

bool arityCompatible(const ObservedOutcomeSet &source,
                     const ObservedOutcomeSet &transformed) {
    return source.arityConsistent && transformed.arityConsistent &&
           source.arity == transformed.arity;
}

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

ObservedOutcomeSet collectOutcomeSet(
    const std::function<std::vector<int64_t>()> &fn, int numRuns,
    int numThreads) {
    // a single caller thread owns the whole sweep; numThreads only sizes the
    // OpenMP team the program runs with, so the harness itself cannot race
    // on the JIT'd module's globals or the OpenMP runtime
    omp_set_num_threads(numThreads);
    ObservedOutcomeSet set;
    std::map<JointOutcome, int64_t> freq;
    for (int i = 0; i < numRuns; ++i) {
        auto res = fn();
        if (set.arityConsistent) {
            if (set.arity == 0)
                set.arity = res.size();
            else if (res.size() != set.arity)
                set.arityConsistent = false;
        }
        ++freq[std::move(res)];
    }
    set.outcomes.reserve(freq.size());
    set.counts.reserve(freq.size());
    for (const auto &[outcome, count] : freq) {
        set.outcomes.push_back(outcome);
        set.counts.push_back(count);
    }
    set.totalRuns = numRuns;
    return set;
}

ObservedOutcomeSet mergeOutcomeSets(const ObservedOutcomeSet &a,
                                    const ObservedOutcomeSet &b) {
    ObservedOutcomeSet out;
    out.totalRuns = a.totalRuns + b.totalRuns;
    out.outcomes.reserve(a.outcomes.size() + b.outcomes.size());
    out.counts.reserve(a.counts.size() + b.counts.size());

    size_t i = 0, j = 0;
    while (i < a.outcomes.size() || j < b.outcomes.size()) {
        if (j >= b.outcomes.size() ||
            (i < a.outcomes.size() && a.outcomes[i] < b.outcomes[j])) {
            out.outcomes.push_back(a.outcomes[i]);
            out.counts.push_back(i < a.counts.size() ? a.counts[i] : 0);
            ++i;
        } else if (i >= a.outcomes.size() ||
                   b.outcomes[j] < a.outcomes[i]) {
            out.outcomes.push_back(b.outcomes[j]);
            out.counts.push_back(j < b.counts.size() ? b.counts[j] : 0);
            ++j;
        } else {
            out.outcomes.push_back(a.outcomes[i]);
            out.counts.push_back((i < a.counts.size() ? a.counts[i] : 0) +
                                 (j < b.counts.size() ? b.counts[j] : 0));
            ++i;
            ++j;
        }
    }

    out.arityConsistent = true;
    for (const auto &jo : out.outcomes) {
        if (out.arity == 0)
            out.arity = jo.size();
        else if (jo.size() != out.arity)
            out.arityConsistent = false;
    }
    return out;
}

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

} // namespace mlir_mr
