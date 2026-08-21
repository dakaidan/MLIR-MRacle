#include "mlir-mracle/oracle/oracle.h"

#include <cstdio>
#include <utility>
#include <vector>

namespace {

using mlir_mracle::CompareResult;
using mlir_mracle::ObservedOutcomeSet;

// single-column joint outcomes (arity 1); entries are {value, count}
ObservedOutcomeSet makeSet(
    const std::vector<std::pair<int64_t, int64_t>> &entries,
    int64_t totalRuns) {
    ObservedOutcomeSet set;
    set.totalRuns = totalRuns;
    for (const auto &[value, count] : entries) {
        set.outcomes.push_back({value});
        set.counts.push_back(count);
    }
    return set;
}

int failures = 0;

void expect(const char *name, const CompareResult &r, bool ok, bool warn) {
    bool pass = r.ok == ok && r.warn == warn;
    std::printf("[%s] %-38s ok=%d warn=%d msg=\"%s\"\n", pass ? "PASS" : "FAIL",
                name, r.ok, r.warn, r.message.c_str());
    if (!pass)
        ++failures;
}

} // namespace

int main() {
    constexpr int kThreshold = 5; // percent

    // source-rare outcome missing in the transformed batch: the source has
    // [2] once in 25000 runs, so the transformed batch is expected to see
    // (1/25000)*5000 = 0.2 occurrences; observing 0 is within the Poisson
    // range -> OK
    {
        auto src = makeSet({{1, 24999}, {2, 1}}, 25000);
        auto tr = makeSet({{1, 5000}}, 5000);
        expect("source-rare missing in transformed",
               mlir_mracle::compareOutcomeSets(src, tr, 2, kThreshold).compare,
               true, false);
    }

    // source stable at 50/50, transformed collapses to a single outcome:
    // both outcomes deviate from the source-predicted rate -> WARN
    {
        auto src = makeSet({{1, 2500}, {2, 2500}}, 5000);
        auto tr = makeSet({{1, 5000}}, 5000);
        expect("inflated shared outcome",
               mlir_mracle::compareOutcomeSets(src, tr, 2, kThreshold).compare,
               true, true);
    }

    // source OK (1%), transformed inflates [2] to 10% of its runs: the fail
    // threshold fires (outcome rare in the source, common in transformed)
    // -> FAIL
    {
        auto src = makeSet({{1, 4950}, {2, 50}}, 5000);
        auto tr = makeSet({{1, 4500}, {2, 500}}, 5000);
        expect("rare-in-source becomes common",
               mlir_mracle::compareOutcomeSets(src, tr, 2, kThreshold).compare,
               false, false);
    }

    // novel outcome (absent from source) once in the transformed batch is
    // within the Poisson range of one expected chance occurrence -> OK
    {
        auto src = makeSet({{1, 5000}}, 5000);
        auto tr = makeSet({{1, 4999}, {2, 1}}, 5000);
        expect("novel single occurrence",
               mlir_mracle::compareOutcomeSets(src, tr, 2, kThreshold).compare,
               true, false);
    }

    // novel outcome above the Poisson bound (50 occurrences vs 1 expected)
    // but below the fail threshold -> WARN
    {
        auto src = makeSet({{1, 5000}}, 5000);
        auto tr = makeSet({{1, 4950}, {2, 50}}, 5000);
        expect("novel rate above bound",
               mlir_mracle::compareOutcomeSets(src, tr, 2, kThreshold).compare,
               true, true);
    }

    // novel outcome at 10% of transformed runs (>= threshold) -> FAIL
    {
        auto src = makeSet({{1, 5000}}, 5000);
        auto tr = makeSet({{1, 4500}, {2, 500}}, 5000);
        expect("novel common outcome",
               mlir_mracle::compareOutcomeSets(src, tr, 2, kThreshold).compare,
               false, false);
    }

    // identical deterministic programs -> OK
    {
        auto src = makeSet({{1, 5000}}, 5000);
        auto tr = makeSet({{1, 5000}}, 5000);
        expect("identical deterministic",
               mlir_mracle::compareOutcomeSets(src, tr, 2, kThreshold).compare,
               true, false);
    }

    // 1-thread determinism path still passes
    {
        auto src = makeSet({{1, 1}}, 1);
        auto tr = makeSet({{1, 1}}, 1);
        expect("single-thread match",
               mlir_mracle::compareOutcomeSets(src, tr, 1, kThreshold).compare,
               true, false);
    }

    // subset: a transformed-only novel outcome at a common rate is a FAIL,
    // since the transformed set must not add outcomes
    {
        auto src = makeSet({{1, 5000}}, 5000);
        auto tr = makeSet({{1, 4500}, {2, 500}}, 5000);
        expect("subset novel common outcome",
               mlir_mracle::compareOutcomeSetsSubset(src, tr, 2, kThreshold)
                   .compare,
               false, false);
    }

    // subset: a transformed-only rare novel outcome stays within the
    // one-chance-occurrence bound -> OK
    {
        auto src = makeSet({{1, 5000}}, 5000);
        auto tr = makeSet({{1, 4999}, {2, 1}}, 5000);
        expect("subset novel single occurrence",
               mlir_mracle::compareOutcomeSetsSubset(src, tr, 2, kThreshold)
                   .compare,
               true, false);
    }

    // superset: extra transformed outcomes are allowed -> OK
    {
        auto src = makeSet({{1, 5000}}, 5000);
        auto tr = makeSet({{1, 4999}, {2, 1}}, 5000);
        expect("superset extra outcome allowed",
               mlir_mracle::compareOutcomeSetsSuperset(src, tr, 2, kThreshold)
                   .compare,
               true, false);
    }

    // superset: a common source outcome missing from the transformed set is
    // a FAIL
    {
        auto src = makeSet({{1, 2500}, {2, 2500}}, 5000);
        auto tr = makeSet({{1, 5000}}, 5000);
        expect("superset common outcome missing",
               mlir_mracle::compareOutcomeSetsSuperset(src, tr, 2, kThreshold)
                   .compare,
               false, false);
    }

    // superset: a rare source outcome missing from the transformed set warns
    // when its absence is statistically significant (e^-50 < 1e-6)
    {
        auto src = makeSet({{1, 4950}, {2, 50}}, 5000);
        auto tr = makeSet({{1, 5000}}, 5000);
        expect("superset rare outcome missing",
               mlir_mracle::compareOutcomeSetsSuperset(src, tr, 2, kThreshold)
                   .compare,
               true, true);
    }

    std::printf("%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILURES",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
