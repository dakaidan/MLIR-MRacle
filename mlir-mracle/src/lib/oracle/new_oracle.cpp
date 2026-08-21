#include "mlir-mracle/oracle/new_oracle.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace mlir_mracle {

namespace {

// p-value below which an outcome count is flagged as above the Poisson
// upper bound of the expected count (mirrors oracle.cpp)
constexpr double kStrongPValue = 1e-6;

// regularised lower incomplete gamma P(a,x) = gamma(a,x)/Gamma(a): series for
// x < a+1, continued fraction for the upper tail otherwise (mirrors oracle.cpp)
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

// P(X > k) for X ~ Poisson(lambda) == scipy.stats.poisson.sf(k, lambda)
double poissonSurvival(int64_t k, double lambda) {
    if (k < 0)
        return 1.0;
    if (lambda <= 0.0)
        return 0.0;
    return lowerIncompleteGamma(static_cast<double>(k) + 1.0, lambda);
}

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

// renders a joint outcome as [a, b, ...] for verdict messages
std::string formatOutcome(const JointOutcome &o) {
    std::string out = "[";
    for (size_t i = 0; i < o.size(); ++i) {
        if (i > 0)
            out += ", ";
        out += std::to_string(o[i]);
    }
    out += "]";
    return out;
}

// joins short deviation clauses into the verdict message
std::string joinReasons(const std::vector<std::string> &reasons) {
    std::string out;
    for (size_t i = 0; i < reasons.size(); ++i) {
        if (i > 0)
            out += "; ";
        out += reasons[i];
    }
    return out;
}

struct StateVerdict {
    bool fail = false;
    bool warn = false;
    bool rerun = false;
};

// novel transformed set (absent from the source): common (>= thresholdPct of
// transformed runs) is a hard behavioural change and fails; a rarer set is
// confirmed by a replay round and then warns, however rare the count is.
// Novel states are exempt from the Poisson upper bound because a state that
// never occurs in the source is unobserved behaviour, not a rate shift.
StateVerdict judgeNovelState(int64_t trCount, int64_t trTotal,
                             int thresholdPct, bool postReruns) {
    if (100.0 * trCount / trTotal >= thresholdPct)
        return {true, false, false};
    if (!postReruns)
        return {false, false, true};
    return {false, true, false};
}

// (position, value) pairs of transformed values that never occur at that
// position in any source outcome
using NovelValue = std::pair<size_t, int64_t>;

// a novel value is a transformed value absent from the source at its
// position; it always warns, even when the outcome sets carrying it are
// rarer than the noise floor, because the value itself is unobserved
// behaviour rather than a rate shift
std::vector<NovelValue> novelValues(const ObservedOutcomeSet &src,
                                    const ObservedOutcomeSet &tr) {
    std::vector<NovelValue> out;
    if (!src.arityConsistent || !tr.arityConsistent ||
        src.arity != tr.arity)
        return out;
    for (size_t i = 0; i < src.arity; ++i) {
        std::set<int64_t> srcVals;
        std::set<int64_t> trVals;
        for (const auto &o : src.outcomes)
            srcVals.insert(o[i]);
        for (const auto &o : tr.outcomes)
            trVals.insert(o[i]);
        for (int64_t v : trVals)
            if (!srcVals.count(v))
                out.push_back({i, v});
    }
    return out;
}

// source state missing from the transformed side: a common source state
// (>= thresholdPct of source runs) fails only when its absence is also
// Poisson-significant; a rarer one warns only when its absence is
// significant, otherwise it is plausibly noise. Before the replay round the
// significant absence triggers a rerun; after it the merged data decides
// between warn and ok.
StateVerdict judgeMissingState(int64_t srcCount, int64_t srcTotal,
                               int64_t trTotal, int thresholdPct,
                               bool postReruns) {
    double expected = (static_cast<double>(srcCount) / srcTotal) * trTotal;
    bool absenceSignificant = std::exp(-expected) < kStrongPValue;
    if (100.0 * srcCount / srcTotal >= thresholdPct && absenceSignificant)
        return {true, false, false};
    if (!absenceSignificant)
        return {false, false, false};
    if (!postReruns)
        return {false, false, true};
    return {false, true, false};
}

// state present on both sides: a rare-in-source/common-in-transformed rate
// crossing fails only when the count also exceeds the Poisson upper bound of
// the source-predicted count; any other count outside the two-sided Poisson
// range warns. The same check runs before and after the replay round, so a
// post-replay verdict is judged on merged data.
StateVerdict judgeSharedState(int64_t srcCount, int64_t srcTotal,
                              int64_t trCount, int64_t trTotal,
                              int thresholdPct) {
    double expected = (static_cast<double>(srcCount) / srcTotal) * trTotal;
    double trPct = 100.0 * trCount / trTotal;
    double pHigh = poissonSurvival(trCount - 1, expected);
    double pLow = 1.0 - poissonSurvival(trCount, expected);
    if (100.0 * srcCount / srcTotal < thresholdPct && trPct >= thresholdPct &&
        pHigh < kStrongPValue)
        return {true, false, false};
    if (pHigh < kStrongPValue || pLow < kStrongPValue)
        return {false, true, false};
    return {false, false, false};
}

} // namespace

NewOracleResult newOracleCompare(const ExecutionResult &exec,
                                 const NewOracleOptions &opts) {
    NewOracleResult out;
    const ObservedOutcomeSet &src = exec.sourceTotal;
    const ObservedOutcomeSet &tr = exec.transformedTotal;

    if (!arityCompatible(src, tr)) {
        out.compare = {false, false,
                       "FAIL: result arity mismatch: source arity " +
                           std::to_string(src.arity) +
                           " vs transformed arity " +
                           std::to_string(tr.arity)};
        return out;
    }

    // the single-thread probes are a determinism check for every relation:
    // both sides must produce exactly the same single outcome
    const ObservedOutcomeSet &srcST = exec.sourceSingleThreadTotal;
    const ObservedOutcomeSet &trST = exec.transformedSingleThreadTotal;
    bool singleMatch = srcST.outcomes.size() == 1 &&
                       trST.outcomes.size() == 1 &&
                       srcST.outcomes.front() == trST.outcomes.front();
    if (!singleMatch) {
        out.compare = {false, false, "FAIL: single-thread determinism mismatch"};
        return out;
    }

    std::vector<JointOutcome> sourceOnly;
    std::vector<JointOutcome> transformedOnly;
    std::set_difference(src.outcomes.begin(), src.outcomes.end(),
                        tr.outcomes.begin(), tr.outcomes.end(),
                        std::back_inserter(sourceOnly));
    std::set_difference(tr.outcomes.begin(), tr.outcomes.end(),
                        src.outcomes.begin(), src.outcomes.end(),
                        std::back_inserter(transformedOnly));

    std::vector<std::string> failReasons;
    std::vector<std::string> warnReasons;
    std::vector<std::string> rerunReasons;
    auto note = [&](const StateVerdict &v, std::string what) {
        if (v.fail)
            failReasons.push_back(std::move(what));
        else if (v.warn)
            warnReasons.push_back(std::move(what));
        else if (v.rerun)
            rerunReasons.push_back(std::move(what));
    };

    auto judgeMissing = [&](const JointOutcome &o) {
        note(judgeMissingState(outcomeCount(src, o), src.totalRuns,
                               tr.totalRuns, opts.thresholdPct,
                               opts.postReruns),
             "missing outcome: " + formatOutcome(o));
    };
    auto judgeNovel = [&](const JointOutcome &o) {
        note(judgeNovelState(outcomeCount(tr, o), tr.totalRuns,
                             opts.thresholdPct, opts.postReruns),
             "novel outcome: " + formatOutcome(o));
    };
    auto judgeShared = [&](const JointOutcome &o) {
        note(judgeSharedState(outcomeCount(src, o), src.totalRuns,
                              outcomeCount(tr, o), tr.totalRuns,
                              opts.thresholdPct),
             "rate shift on shared outcome: " + formatOutcome(o));
    };
    auto judgeNovelValue = [&](const NovelValue &nv) {
        note({false, true, false},
             "novel value: var" + std::to_string(nv.first) + "=" +
                 std::to_string(nv.second));
    };

    switch (opts.relation) {
    case OutcomeRelation::Subset:
        for (const auto &o : transformedOnly)
            judgeNovel(o);
        for (const auto &nv : novelValues(src, tr))
            judgeNovelValue(nv);
        break;
    case OutcomeRelation::Superset:
        for (const auto &o : sourceOnly)
            judgeMissing(o);
        break;
    default:
        for (const auto &o : sourceOnly)
            judgeMissing(o);
        for (const auto &o : transformedOnly)
            judgeNovel(o);
        for (const auto &nv : novelValues(src, tr))
            judgeNovelValue(nv);
        for (const auto &o : src.outcomes)
            if (outcomeCount(tr, o) > 0)
                judgeShared(o);
        break;
    }

    // the message names every warn/fail-worthy set, including rate-shifted
    // shared sets, so a hard fail also reports the states that merely warn
    if (!failReasons.empty()) {
        std::vector<std::string> all = failReasons;
        all.insert(all.end(), warnReasons.begin(), warnReasons.end());
        out.compare = {false, false, joinReasons(all)};
        return out;
    }
    if (!rerunReasons.empty()) {
        std::vector<std::string> all = rerunReasons;
        all.insert(all.end(), warnReasons.begin(), warnReasons.end());
        out.compare = {true, false,
                       "rare outcome present; replay pending: " +
                           joinReasons(all)};
        out.needsRerun = true;
        return out;
    }
    if (!warnReasons.empty()) {
        out.compare = {true, true, joinReasons(warnReasons)};
        return out;
    }
    out.compare = {true, false, ""};
    return out;
}

} // namespace mlir_mracle
