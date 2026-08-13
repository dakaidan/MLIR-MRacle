#include "mlir-mr/oracle/oracle.h"

#include <algorithm>
#include <iterator>
#include <map>
#include <omp.h>
#include <vector>
#include <cmath>

namespace mlir_mr {

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

OutcomeSetResult compareOutcomeSets(const ObservedOutcomeSet &source,
                                    const ObservedOutcomeSet &transformed,
                                    int numThreads, int thresholdPct) {
    OutcomeSetResult result;
    result.source = source;
    result.transformed = transformed;

    if (!source.arityConsistent || !transformed.arityConsistent ||
        source.arity != transformed.arity) {
        result.compare = {false, false, "result arity mismatch"};
        return result;
    }

    std::vector<JointOutcome> sourceOnly;
    std::vector<JointOutcome> transformedOnly;
    std::set_difference(source.outcomes.begin(), source.outcomes.end(),
                        transformed.outcomes.begin(), transformed.outcomes.end(),
                        std::back_inserter(sourceOnly));
    std::set_difference(transformed.outcomes.begin(), transformed.outcomes.end(),
                        source.outcomes.begin(), source.outcomes.end(),
                        std::back_inserter(transformedOnly));

    if (numThreads == 1) {
        bool nondet =
            source.outcomes.size() > 1 || transformed.outcomes.size() > 1;
        if (!sourceOnly.empty() || !transformedOnly.empty()) {
            result.compare = {false, false,
                              nondet ? "non-deterministic with 1 thread; result mismatch"
                                     : "result mismatch"};
        } else {
            result.compare = {true, nondet,
                              nondet ? "non-deterministic with 1 thread" : ""};
        }
        return result;
    }

    if (!transformedOnly.empty()) {
        bool anyWarn = false;
        bool anyFail = false;
        for (const auto &outcome : transformedOnly) {
            auto it = std::lower_bound(transformed.outcomes.begin(),
                                       transformed.outcomes.end(), outcome);
            int64_t count = 0;
            if (it != transformed.outcomes.end() && *it == outcome) {
                size_t idx = std::distance(transformed.outcomes.begin(), it);
                if (idx < transformed.counts.size())
                    count = transformed.counts[idx];
            }
            double pct = 100.0 * count /
                         std::max<int64_t>(1, transformed.totalRuns);
            if (pct < thresholdPct)
                anyWarn = true;
            else
                anyFail = true;
        }
        if (anyFail) {
            result.compare = {false, false, "behavioural change detected"};
        } else {
            result.compare = {true, true,
                              "rare transformed-only outcome(s) not found in source"};
        }
        return result;
    }

    if (!sourceOnly.empty()) {
        result.compare = {true, true, "source-only outcomes observed"};
    } else {
        result.compare = {true, false, ""};
    }
    return result;
}

} // namespace mlir_mr
