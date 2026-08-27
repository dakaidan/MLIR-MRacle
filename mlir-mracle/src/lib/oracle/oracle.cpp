#include "mlir-mracle/oracle/oracle.h"
#include "mlir-mracle/oracle/statistics.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace mlir_mracle {

namespace {

using oracle_detail::arityCompatible;
using oracle_detail::kStrongPValue;
using oracle_detail::outcomeCount;
using oracle_detail::poissonSurvival;

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

// called on outcomes that are present in the transformed set but absent from the source
// if count >= thresholdPct of transformed runs, it is a hard fail - behavioural change
// if count < thresholdPct, it is a warn - inconclusive but possible behavioral change
// if warn, rerun to gather more data and confirm the verdict or until run budget is reached
StateVerdict judgeNovelState(int64_t trCount, int64_t trTotal,
                             int thresholdPct, bool postReruns) {
    if (100.0 * trCount / trTotal >= thresholdPct)
        return {true, false, false};
    if (!postReruns)
        return {false, false, true};
    return {false, true, false};
}

// (position, value) pairs of transformed values that never occur at that position in any source outcome
using NovelValue = std::pair<size_t, int64_t>;

// a novel value is a value that never appears at its position in any source outcome
// it is a warn no matter what, as that position has never seen this value before
//
// key difference:
// - novel value: value absent from a specific position in every source outcome
// - novel outcome: values are individually possible, just not this specific combination
std::vector<NovelValue> novelValues(const ObservedOutcomeSet &src,
                                    const ObservedOutcomeSet &tr) {
    std::vector<NovelValue> out;

    if (!src.arityConsistent || !tr.arityConsistent || src.arity != tr.arity)
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

// function for judging a state missing from the source set
// performs a poisson test on the source-predicted count of the state in the transformed set
// done to guard against false positives from rare states that are not actually missing
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

// function for judging a state that is present in both the source and transformed sets
// performs a poisson test on the source-predicted count of the state in the transformed set
// done to check for rate shifts on shared states, which are a hard fail if significant
StateVerdict judgeSharedState(int64_t srcCount, int64_t srcTotal,
                              int64_t trCount, int64_t trTotal,
                              int thresholdPct) {
    double expected = (static_cast<double>(srcCount) / srcTotal) * trTotal;
    double trPct = 100.0 * trCount / trTotal;
    double pHigh = poissonSurvival(trCount - 1, expected);
    double pLow = 1.0 - poissonSurvival(trCount, expected);

    if (100.0 * srcCount / srcTotal < thresholdPct && trPct >= thresholdPct && pHigh < kStrongPValue)
        return {true, false, false};

    if (pHigh < kStrongPValue || pLow < kStrongPValue)
        return {false, true, false};

    return {false, false, false};
}

} // namespace

// Main oracle comparison function
OracleResult oracleCompare(const ExecutionResult &exec,
                           const OracleOptions &opts) {
    OracleResult out;
    const ObservedOutcomeSet &src = exec.sourceTotal;
    const ObservedOutcomeSet &tr = exec.transformedTotal;

    // arity mismatch is a hard fail;
    // the transformed program is producing a different number of results
    if (!arityCompatible(src, tr)) {
        out.compare.issues.push_back(
            {IssueSeverity::Fail, "",
             "result arity mismatch: source arity " +
                 std::to_string(src.arity) +
                 " vs transformed arity " + std::to_string(tr.arity)});
        return out;
    }

    // the single-thread probes are a determinism check for every relation
    // both sides must produce exactly the same single outcome
    const ObservedOutcomeSet &srcST = exec.sourceSingleThreadTotal;
    const ObservedOutcomeSet &trST = exec.transformedSingleThreadTotal;
    bool singleMatch = srcST.outcomes.size() == 1 &&
                       trST.outcomes.size() == 1 &&
                       srcST.outcomes.front() == trST.outcomes.front();
                       
    if (!singleMatch) {
        out.compare.issues.push_back(
            {IssueSeverity::Fail, "", "single-thread determinism mismatch"});
        return out;
    }

    std::vector<JointOutcome> sourceOnly;
    std::vector<JointOutcome> transformedOnly;

    // compute the set differences between the source and transformed outcome sets
    std::set_difference(src.outcomes.begin(), src.outcomes.end(),
                        tr.outcomes.begin(), tr.outcomes.end(),
                        std::back_inserter(sourceOnly));
    std::set_difference(tr.outcomes.begin(), tr.outcomes.end(),
                        src.outcomes.begin(), src.outcomes.end(),
                        std::back_inserter(transformedOnly));

    std::vector<std::string> warnReasons;
    std::vector<std::string> rerunReasons;
    std::vector<VerdictIssue> failIssues;
    std::vector<VerdictIssue> warnIssues;

    // helper function for recording a verdict on a state
    // and adding the appropriate message to the verdict
    auto note = [&](const StateVerdict &v, std::string what,
                    VerdictIssue issue) {
        if (v.fail) {
            issue.severity = IssueSeverity::Fail;
            failIssues.push_back(std::move(issue));
        } else if (v.warn) {
            issue.severity = IssueSeverity::Warn;
            warnIssues.push_back(std::move(issue));
            warnReasons.push_back(std::move(what));
        } else if (v.rerun) {
            rerunReasons.push_back(std::move(what));
        }
    };

    // helper functions for judging the different types of states
    auto judgeMissing = [&](const JointOutcome &o) {
        std::string outcome = formatOutcome(o);
        note(judgeMissingState(outcomeCount(src, o), src.totalRuns,
                               tr.totalRuns, opts.thresholdPct,
                               opts.postReruns),
             "missing outcome: " + outcome,
             {IssueSeverity::Fail, outcome, "missing outcome"});
    };

    auto judgeNovel = [&](const JointOutcome &o) {
        std::string outcome = formatOutcome(o);
        note(judgeNovelState(outcomeCount(tr, o), tr.totalRuns,
                             opts.thresholdPct, opts.postReruns),
             "novel outcome: " + outcome,
             {IssueSeverity::Fail, outcome, "novel outcome"});
    };

    auto judgeShared = [&](const JointOutcome &o) {
        std::string outcome = formatOutcome(o);
        note(judgeSharedState(outcomeCount(src, o), src.totalRuns,
                              outcomeCount(tr, o), tr.totalRuns,
                              opts.thresholdPct),
             "rate shift on shared outcome: " + outcome,
             {IssueSeverity::Fail, outcome, "rate shift on shared outcome"});
    };

    auto judgeNovelValue = [&](const NovelValue &nv) {
        std::string value = "var" + std::to_string(nv.first) + "=" +
                            std::to_string(nv.second);
        note({false, true, false}, "novel value: " + value,
             {IssueSeverity::Warn, value, "novel value"});
    };

    switch (opts.relation) {
        // the equality relation checks for missing, novel, and shared states
        case OutcomeRelation::Subset:
            for (const auto &o : transformedOnly)
                judgeNovel(o);

            for (const auto &nv : novelValues(src, tr))
                judgeNovelValue(nv);
            break;
        // the superset relation checks for missing states only
        case OutcomeRelation::Superset:
            for (const auto &o : sourceOnly)
                judgeMissing(o);
            break;
        // the subset relation checks for novel and shared states only
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
    if (!failIssues.empty()) {
        out.compare.issues = failIssues;
        out.compare.issues.insert(out.compare.issues.end(),
                                  warnIssues.begin(), warnIssues.end());
        return out;
    }

    if (!rerunReasons.empty()) {
        std::vector<std::string> all = rerunReasons;
        all.insert(all.end(), warnReasons.begin(), warnReasons.end());
        out.compare.note =
            "rare outcome present; replay pending: " + joinReasons(all);
        out.needsRerun = true;
        return out;
    }
    
    if (!warnReasons.empty()) {
        out.compare.issues = warnIssues;
        return out;
    }
    return out;
}

} // namespace mlir_mracle
