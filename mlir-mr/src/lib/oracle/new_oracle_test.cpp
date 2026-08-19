// Regression tests for the new-oracle judges. The .cpp is included directly
// (instead of linking the library) so the static judge functions under test
// are reachable; the library's own copy of new_oracle.cpp is therefore not
// linked into this executable.
#include "mlir-mr/oracle/new_oracle.h"
#include "new_oracle.cpp"

#include <cstdio>
#include <utility>
#include <vector>
#include <set>

namespace {

using mlir_mr::ExecutionResult;
using mlir_mr::NewOracleOptions;
using mlir_mr::NewOracleResult;
using mlir_mr::ObservedOutcomeSet;
using mlir_mr::OutcomeRelation;
using mlir_mr::StateVerdict;

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

ExecutionResult makeExec(const ObservedOutcomeSet &src,
                         const ObservedOutcomeSet &tr) {
    ExecutionResult exec;
    exec.sourceTotal = src;
    exec.transformedTotal = tr;
    ObservedOutcomeSet st = makeSet({{1, 1}}, 1);
    exec.sourceSingleThreadTotal = st;
    exec.transformedSingleThreadTotal = st;
    return exec;
}

int failures = 0;

void expectVerdict(const char *name, const StateVerdict &v, bool fail,
                   bool warn, bool rerun) {
    bool pass = v.fail == fail && v.warn == warn && v.rerun == rerun;
    std::printf("[%s] %-42s fail=%d warn=%d rerun=%d\n",
                pass ? "PASS" : "FAIL", name, v.fail, v.warn, v.rerun);
    if (!pass)
        ++failures;
}

void expectResult(const char *name, const NewOracleResult &r, bool ok,
                  bool warn, bool needsRerun) {
    bool pass =
        r.compare.ok == ok && r.compare.warn == warn && r.needsRerun == needsRerun;
    std::printf("[%s] %-42s ok=%d warn=%d rerun=%d msg=\"%s\"\n",
                pass ? "PASS" : "FAIL", name, r.compare.ok, r.compare.warn,
                r.needsRerun, r.compare.message.c_str());
    if (!pass)
        ++failures;
}

} // namespace

int main() {
    constexpr int kThreshold = 5; // percent

    // --- novel state judges ---

    // the value never occurred in the source, so it is replayed pre-replay
    // to confirm it; the noise floor keeps the weak count from failing:
    // regression for the [1,1] count-2 false FAIL
    expectVerdict("novel weak within noise floor",
                  mlir_mr::judgeNovelState(2, 25000, kThreshold, false),
                  false, false, true);

    // significant but rare novel count asks for a replay before the round
    expectVerdict("novel significant rare pre-replay",
                  mlir_mr::judgeNovelState(74, 25000, kThreshold, false),
                  false, false, true);

    // still rare after the replay round warns, never fails
    expectVerdict("novel significant rare post-replay",
                  mlir_mr::judgeNovelState(74, 25000, kThreshold, true),
                  false, true, false);

    // significant and common (>= thresholdPct of transformed runs) fails
    expectVerdict("novel strong common",
                  mlir_mr::judgeNovelState(2000, 25000, kThreshold, false),
                  true, false, false);

    // the noise floor scales with the batch: the same rate at 10x the runs
    // sits inside the null, so small-batch noise does not fail; post-replay
    // the novel value still warns on its own
    expectVerdict("novel noise floor scales with N",
                  mlir_mr::judgeNovelState(100, 250000, kThreshold, true),
                  false, true, false);

    // --- shared state judges ---

    // rare-in-source (1%) common-in-transformed (10%) with a count beyond the
    // Poisson upper bound of the expected count -> fail
    expectVerdict("shared rate crossing significant",
                  mlir_mr::judgeSharedState(50, 5000, 500, 5000, kThreshold),
                  true, false, false);

    // the same rate crossing without statistical significance (4.9% -> 5.0%
    // over 100k runs) is indistinguishable from noise -> ok
    expectVerdict("shared rate crossing not significant",
                  mlir_mr::judgeSharedState(4900, 100000, 5000, 100000, kThreshold),
                  false, false, false);

    // count outside the two-sided Poisson range without a threshold crossing
    // warns
    expectVerdict("shared inflated without crossing",
                  mlir_mr::judgeSharedState(2500, 5000, 4750, 5000, kThreshold),
                  false, true, false);

    // stable equal rates -> ok
    expectVerdict("shared stable", mlir_mr::judgeSharedState(2500, 5000, 2500, 5000,
                                                    kThreshold),
                  false, false, false);

    // --- missing state judges ---

    // common source outcome (50%) absent from a transformed batch large
    // enough that the absence is Poisson-significant -> fail
    expectVerdict("missing common significant",
                  mlir_mr::judgeMissingState(2500, 5000, 5000, kThreshold, true),
                  true, false, false);

    // common source outcome absent from a tiny transformed batch is within
    // the Poisson range of the predicted count -> ok
    expectVerdict("missing common not significant",
                  mlir_mr::judgeMissingState(2500, 5000, 2, kThreshold, true),
                  false, false, false);

    // rare source outcome (1%) missing from a batch that should contain ~50
    // of them warns after the replay round
    expectVerdict("missing rare significant post",
                  mlir_mr::judgeMissingState(50, 5000, 5000, kThreshold, true),
                  false, true, false);

    // the same significant absence pre-replay triggers a rerun
    expectVerdict("missing rare significant pre",
                  mlir_mr::judgeMissingState(50, 5000, 5000, kThreshold, false),
                  false, false, true);

    // rare source outcome missing from a batch where ~0.2 were expected is
    // plausibly noise -> ok
    expectVerdict("missing rare not significant",
                  mlir_mr::judgeMissingState(1, 25000, 5000, kThreshold, true),
                  false, false, false);

    // --- newOracleCompare end to end ---

    NewOracleOptions opts;
    opts.thresholdPct = kThreshold;
    opts.relation = OutcomeRelation::Equality;

    // regression: source has four outcomes at 25% each; the transformed side
    // adds a novel [2,2] outcome at count 2. The value itself never occurred
    // in the source, so pre-replay the weak count still asks for a replay
    // round, and the post-replay verdict warns.
    {
        auto src = makeSet({{0, 6250}, {1, 6250}, {2, 6250}, {3, 6250}},
                           25000);
        auto tr = makeSet({{0, 6250}, {1, 6250}, {2, 6250}, {3, 6248},
                           {4, 2}},
                          25000);
        auto exec = makeExec(src, tr);
        opts.postReruns = false;
        expectResult("equality weak novel pre-replay",
                     newOracleCompare(exec, opts), true, false, true);
        opts.postReruns = true;
        expectResult("equality weak novel post-replay",
                     newOracleCompare(exec, opts), true, true, false);
    }

    // a significant-but-rare novel count asks for a replay pre-replay and
    // warns post-replay
    {
        auto src = makeSet({{1, 25000}}, 25000);
        auto tr = makeSet({{1, 24926}, {2, 74}}, 25000);
        auto exec = makeExec(src, tr);
        opts.postReruns = false;
        expectResult("equality significant novel rare pre",
                     newOracleCompare(exec, opts), true, false, true);
        opts.postReruns = true;
        expectResult("equality significant novel rare post",
                     newOracleCompare(exec, opts), true, true, false);
    }

    // a strong novel count that is also common (>= thresholdPct) fails
    // before and after the replay round
    {
        auto src = makeSet({{1, 23000}}, 25000);
        auto tr = makeSet({{1, 23000}, {2, 2000}}, 25000);
        auto exec = makeExec(src, tr);
        opts.postReruns = false;
        expectResult("equality strong novel", newOracleCompare(exec, opts),
                     false, false, false);
    }

    // a common source outcome missing from a large transformed batch fails
    {
        auto src = makeSet({{1, 12500}, {2, 12500}}, 25000);
        auto tr = makeSet({{1, 25000}}, 25000);
        auto exec = makeExec(src, tr);
        opts.postReruns = true;
        expectResult("equality common missing", newOracleCompare(exec, opts),
                     false, false, false);
    }

    // single-thread determinism mismatch is a hard fail for every relation
    {
        auto src = makeSet({{1, 25000}}, 25000);
        auto tr = makeSet({{1, 25000}}, 25000);
        auto exec = makeExec(src, tr);
        exec.transformedSingleThreadTotal = makeSet({{2, 1}}, 1);
        opts.postReruns = true;
        expectResult("single-thread mismatch", newOracleCompare(exec, opts),
                     false, false, false);
    }

    // subset relation: transformed must not add outcomes
    {
        opts.relation = OutcomeRelation::Subset;
        auto src = makeSet({{1, 5000}}, 5000);
        auto tr = makeSet({{1, 4500}, {2, 500}}, 5000);
        auto exec = makeExec(src, tr);
        expectResult("subset strong novel", newOracleCompare(exec, opts),
                     false, false, false);

        auto tr2 = makeSet({{1, 4998}, {2, 2}}, 5000);
        auto exec2 = makeExec(src, tr2);
        opts.postReruns = true;
        expectResult("subset weak novel post-replay",
                     newOracleCompare(exec2, opts), true, true, false);
    }

    // superset relation: transformed must keep every source outcome
    {
        opts.relation = OutcomeRelation::Superset;
        auto src = makeSet({{1, 2500}, {2, 2500}}, 5000);
        auto tr = makeSet({{1, 5000}}, 5000);
        auto exec = makeExec(src, tr);
        expectResult("superset common missing", newOracleCompare(exec, opts),
                     false, false, false);

        auto src2 = makeSet({{1, 24999}, {2, 1}}, 25000);
        auto tr2 = makeSet({{1, 25000}}, 25000);
        auto exec2 = makeExec(src2, tr2);
        opts.postReruns = true;
        expectResult("superset rare missing not significant",
                     newOracleCompare(exec2, opts), true, false, false);
    }

    // equality: a threshold-crossing shared outcome that is not
    // Poisson-significant is ok
    {
        opts.relation = OutcomeRelation::Equality;
        auto src = makeSet({{1, 95100}, {2, 4900}}, 100000);
        auto tr = makeSet({{1, 95000}, {2, 5000}}, 100000);
        auto exec = makeExec(src, tr);
        opts.postReruns = true;
        expectResult("equality crossing not significant",
                     newOracleCompare(exec, opts), true, false, false);
    }

    std::printf("%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILURES",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
