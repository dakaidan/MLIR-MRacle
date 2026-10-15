// Regression tests for the default oracle judges. The .cpp is included
// directly (instead of linking the library) so the static judge functions
// under test are reachable; the library's own copy of oracle.cpp is
// therefore not linked into this executable.
#include "mlir-mracle/oracle/oracle.h"
#include "oracle.cpp"

#include <cstdio>
#include <utility>
#include <vector>
#include <set>
#include <algorithm>

namespace {

using mlir_mracle::ExecutionResult;
using mlir_mracle::OracleOptions;
using mlir_mracle::OracleResult;
using mlir_mracle::ObservedOutcomeSet;
using mlir_mracle::OutcomeRelation;
using mlir_mracle::StateVerdict;

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

void expectResult(const char *name, const OracleResult &r, bool ok,
                  bool warn, bool needsRerun) {
    bool pass = r.compare.ok() == ok && r.compare.warn() == warn &&
                r.needsRerun == needsRerun;
    std::printf("[%s] %-42s ok=%d warn=%d rerun=%d msg=\"%s\"\n",
                pass ? "PASS" : "FAIL", name, r.compare.ok(), r.compare.warn(),
                r.needsRerun, r.compare.message().c_str());
    if (!pass)
        ++failures;
}

void expectMessageContains(const char *name, const std::string &msg,
                           const char *needle) {
    bool pass = msg.find(needle) != std::string::npos;
    std::printf("[%s] %-42s msg=\"%s\" needle=\"%s\"\n",
                pass ? "PASS" : "FAIL", name, msg.c_str(), needle);
    if (!pass)
        ++failures;
}

void expectIssue(const char *name, const OracleResult &r,
                 mlir_mracle::IssueSeverity severity, const char *reason,
                 const char *outcome) {
    bool pass = false;
    for (const auto &issue : r.compare.issues)
        if (issue.severity == severity && issue.reason == reason &&
            (outcome == nullptr || issue.outcome == outcome)) {
            pass = true;
            break;
        }
    std::printf("[%s] %-42s issues=%zu status=%s reason=\"%s\" outcome=\"%s\"\n",
                pass ? "PASS" : "FAIL", name, r.compare.issues.size(),
                mlir_mracle::issueSeverityToString(severity).c_str(), reason,
                outcome == nullptr ? "" : outcome);
    if (!pass)
        ++failures;
}

// FAIL issues must precede WARN issues in the structured list
void expectFailIssuesFirst(const char *name, const OracleResult &r) {
    bool pass = true;
    bool seenWarn = false;
    for (const auto &issue : r.compare.issues) {
        if (issue.severity == mlir_mracle::IssueSeverity::Warn)
            seenWarn = true;
        else if (seenWarn)
            pass = false;
    }
    std::printf("[%s] %-42s issues=%zu\n", pass ? "PASS" : "FAIL", name,
                r.compare.issues.size());
    if (!pass)
        ++failures;
}

void expectNovelValues(const char *name,
                       const std::vector<std::pair<size_t, int64_t>> &actual,
                       const std::vector<std::pair<size_t, int64_t>> &expected) {
    bool pass = actual == expected;
    std::printf("[%s] %-42s novel_values=%zu\n", pass ? "PASS" : "FAIL",
                name, actual.size());
    if (!pass)
        ++failures;
}

} // namespace

int main() {
    constexpr int kThreshold = 5; // percent

    // --- novel set judges ---

    // a novel set is exempt from the Poisson bound: any rare count is
    // confirmed by a replay round pre-replay and then warns, never fails
    expectVerdict("novel rare pre-replay",
                  mlir_mracle::judgeNovelState(2, 25000, kThreshold, false),
                  false, false, true);

    // the rare novel set warns after the replay round
    expectVerdict("novel rare post-replay",
                  mlir_mracle::judgeNovelState(2, 25000, kThreshold, true),
                  false, true, false);

    // a larger-but-still-rare novel count behaves the same
    expectVerdict("novel significant rare pre-replay",
                  mlir_mracle::judgeNovelState(74, 25000, kThreshold, false),
                  false, false, true);
    expectVerdict("novel significant rare post-replay",
                  mlir_mracle::judgeNovelState(74, 25000, kThreshold, true),
                  false, true, false);

    // common (>= thresholdPct of transformed runs) fails outright
    expectVerdict("novel common",
                  mlir_mracle::judgeNovelState(2000, 25000, kThreshold, false),
                  true, false, false);

    // exactly at the threshold counts as common
    expectVerdict("novel common boundary",
                  mlir_mracle::judgeNovelState(12500, 250000, kThreshold, false),
                  true, false, false);

    // --- novel value detection ---

    {
        auto src = makeSet({{1, 24000}, {2, 1000}}, 25000);
        auto tr = makeSet({{1, 24500}, {2, 500}}, 25000);
        src.arity = 1;
        tr.arity = 1;
        expectNovelValues("no novel value", mlir_mracle::novelValues(src, tr), {});
    }

    // 3 at index 0 never occurs in the source, so it is a novel value even
    // though the outcome set carrying it is rarer than the noise floor
    {
        auto src = makeSet({{1, 24000}, {2, 1000}}, 25000);
        auto tr = makeSet({{1, 24500}, {3, 500}}, 25000);
        src.arity = 1;
        tr.arity = 1;
        expectNovelValues("novel value detected", mlir_mracle::novelValues(src, tr),
                          {{0, 3}});
    }

    // position matters: 5 is novel only at index 1, 1 at index 0 is shared
    {
        ObservedOutcomeSet src, tr;
        src.arity = tr.arity = 2;
        src.arityConsistent = tr.arityConsistent = true;
        src.totalRuns = tr.totalRuns = 25000;
        src.outcomes = {{1, 2}, {3, 0}};
        src.counts = {12500, 12500};
        tr.outcomes = {{1, 5}, {3, 0}};
        tr.counts = {12500, 12500};
        expectNovelValues("novel value by position",
                          mlir_mracle::novelValues(src, tr), {{1, 5}});
    }

    // --- shared state judges ---

    // rare-in-source (1%) common-in-transformed (10%) with a count beyond the
    // Poisson upper bound of the expected count -> fail
    expectVerdict("shared rate crossing significant",
                  mlir_mracle::judgeSharedState(50, 5000, 500, 5000, kThreshold),
                  true, false, false);

    // the same rate crossing without statistical significance (4.9% -> 5.0%
    // over 100k runs) is indistinguishable from noise -> ok
    expectVerdict("shared rate crossing not significant",
                  mlir_mracle::judgeSharedState(4900, 100000, 5000, 100000, kThreshold),
                  false, false, false);

    // count outside the two-sided Poisson range without a threshold crossing
    // warns
    expectVerdict("shared inflated without crossing",
                  mlir_mracle::judgeSharedState(2500, 5000, 4750, 5000, kThreshold),
                  false, true, false);

    // stable equal rates -> ok
    expectVerdict("shared stable", mlir_mracle::judgeSharedState(2500, 5000, 2500, 5000,
                                                    kThreshold),
                  false, false, false);

    // --- missing state judges ---

    // common source outcome (50%) absent from a transformed batch large
    // enough that the absence is Poisson-significant -> fail
    expectVerdict("missing common significant",
                  mlir_mracle::judgeMissingState(2500, 5000, 5000, kThreshold, true),
                  true, false, false);

    // common source outcome absent from a tiny transformed batch is within
    // the Poisson range of the predicted count -> ok
    expectVerdict("missing common not significant",
                  mlir_mracle::judgeMissingState(2500, 5000, 2, kThreshold, true),
                  false, false, false);

    // rare source outcome (1%) missing from a batch that should contain ~50
    // of them warns after the replay round
    expectVerdict("missing rare significant post",
                  mlir_mracle::judgeMissingState(50, 5000, 5000, kThreshold, true),
                  false, true, false);

    // the same significant absence pre-replay triggers a rerun
    expectVerdict("missing rare significant pre",
                  mlir_mracle::judgeMissingState(50, 5000, 5000, kThreshold, false),
                  false, false, true);

    // rare source outcome missing from a batch where ~0.2 were expected is
    // plausibly noise -> ok
    expectVerdict("missing rare not significant",
                  mlir_mracle::judgeMissingState(1, 25000, 5000, kThreshold, true),
                  false, false, false);

    // --- oracleCompare end to end ---


    OracleOptions opts;
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
                     oracleCompare(exec, opts), true, false, true);
        opts.postReruns = true;
        auto r = oracleCompare(exec, opts);
        expectResult("equality weak novel post-replay", r, true, true, false);
        expectIssue("equality weak novel post-replay issue", r,
                    mlir_mracle::IssueSeverity::Warn, "novel outcome", "[4]");
    }

    // a significant-but-rare novel count asks for a replay pre-replay and
    // warns post-replay
    {
        auto src = makeSet({{1, 25000}}, 25000);
        auto tr = makeSet({{1, 24926}, {2, 74}}, 25000);
        auto exec = makeExec(src, tr);
        opts.postReruns = false;
        expectResult("equality significant novel rare pre",
                     oracleCompare(exec, opts), true, false, true);
        opts.postReruns = true;
        expectResult("equality significant novel rare post",
                     oracleCompare(exec, opts), true, true, false);
    }

    // a strong novel count that is also common (>= thresholdPct) fails
    // before and after the replay round
    {
        auto src = makeSet({{1, 23000}}, 25000);
        auto tr = makeSet({{1, 23000}, {2, 2000}}, 25000);
        auto exec = makeExec(src, tr);
        opts.postReruns = false;
        expectResult("equality strong novel", oracleCompare(exec, opts),
                     false, false, false);
    }

    // a novel value (3 at index 0, never in the source) warns directly even
    // though its outcome set is rarer than the noise floor, and the message
    // names the value instead of a generic rate shift
    {
        auto src = makeSet({{1, 24998}, {2, 2}}, 25000);
        auto tr = makeSet({{1, 24996}, {2, 2}, {3, 2}}, 25000);
        auto exec = makeExec(src, tr);
        exec.sourceTotal.arity = 1;
        exec.transformedTotal.arity = 1;
        opts.postReruns = true;
        auto r = oracleCompare(exec, opts);
        expectResult("equality novel value post-replay", r, true, true, false);
        expectMessageContains("equality novel value message",
                              r.compare.message(), "novel value: var0=3");
        expectIssue("equality novel value issue", r,
                    mlir_mracle::IssueSeverity::Warn, "novel value", "var0=3");
    }

    // pre-replay the rare novel set carrying the novel value still asks for
    // a replay round
    {
        auto src = makeSet({{1, 24998}, {2, 2}}, 25000);
        auto tr = makeSet({{1, 24996}, {2, 2}, {3, 2}}, 25000);
        auto exec = makeExec(src, tr);
        exec.sourceTotal.arity = 1;
        exec.transformedTotal.arity = 1;
        opts.postReruns = false;
        expectResult("equality novel value pre-replay",
                     oracleCompare(exec, opts), true, false, true);
    }

    // every warn/fail-worthy set is named in the message: a Poisson-
    // significant rate crossing fails, and the rare novel set warns; both
    // reasons are displayed together
    {
        auto src = makeSet({{1, 24998}, {2, 1}}, 25000);
        auto tr = makeSet({{1, 19999}, {2, 5000}, {3, 1}}, 25000);
        auto exec = makeExec(src, tr);
        exec.sourceTotal.arity = 1;
        exec.transformedTotal.arity = 1;
        opts.postReruns = true;
        auto r = oracleCompare(exec, opts);
        expectResult("equality fail plus warn", r, false, false, false);
        expectMessageContains("equality fail plus warn rate shift",
                              r.compare.message(),
                              "rate shift on shared outcome: [2]");
        expectMessageContains("equality fail plus warn novel",
                              r.compare.message(), "novel outcome: [3]");
        expectIssue("equality fail plus warn rate shift issue", r,
                    mlir_mracle::IssueSeverity::Fail,
                    "rate shift on shared outcome", "[2]");
        expectIssue("equality fail plus warn novel issue", r,
                    mlir_mracle::IssueSeverity::Warn, "novel outcome", "[3]");
        expectFailIssuesFirst("equality fail plus warn issue order", r);
    }

    // the pre-replay pending message lists warn-worthy sets too
    {
        auto src = makeSet({{1, 22500}, {2, 2500}}, 25000);
        auto tr = makeSet({{1, 20998}, {2, 4000}, {3, 2}}, 25000);
        auto exec = makeExec(src, tr);
        exec.sourceTotal.arity = 1;
        exec.transformedTotal.arity = 1;
        opts.postReruns = false;
        auto r = oracleCompare(exec, opts);
        expectResult("equality pending lists warn", r, true, false, true);
        expectMessageContains("equality pending lists warn message",
                              r.compare.message(),
                              "rate shift on shared outcome: [2]");
    }

    // a rate-shifted shared set warns and is named in the message
    {
        auto src = makeSet({{1, 4750}, {2, 250}}, 5000);
        auto tr = makeSet({{1, 250}, {2, 4750}}, 5000);
        auto exec = makeExec(src, tr);
        exec.sourceTotal.arity = 1;
        exec.transformedTotal.arity = 1;
        opts.postReruns = true;
        auto r = oracleCompare(exec, opts);
        expectResult("equality shared rate warn", r, true, true, false);
        expectMessageContains("equality shared rate warn message",
                              r.compare.message(),
                              "rate shift on shared outcome: [2]");
    }

    // a common source outcome missing from a large transformed batch fails
    {
        auto src = makeSet({{1, 12500}, {2, 12500}}, 25000);
        auto tr = makeSet({{1, 25000}}, 25000);
        auto exec = makeExec(src, tr);
        opts.postReruns = true;
        expectResult("equality common missing", oracleCompare(exec, opts),
                     false, false, false);
    }

    // single-thread determinism mismatch is a hard fail for every relation
    {
        auto src = makeSet({{1, 25000}}, 25000);
        auto tr = makeSet({{1, 25000}}, 25000);
        auto exec = makeExec(src, tr);
        exec.transformedSingleThreadTotal = makeSet({{2, 1}}, 1);
        opts.postReruns = true;
        auto r = oracleCompare(exec, opts);
        expectResult("single-thread mismatch", r, false, false, false);
        expectIssue("single-thread mismatch issue", r,
                    mlir_mracle::IssueSeverity::Fail,
                    "single-thread determinism mismatch", nullptr);
    }

    // subset relation: transformed must not add outcomes
    {
        opts.relation = OutcomeRelation::Subset;
        auto src = makeSet({{1, 5000}}, 5000);
        auto tr = makeSet({{1, 4500}, {2, 500}}, 5000);
        auto exec = makeExec(src, tr);
        expectResult("subset strong novel", oracleCompare(exec, opts),
                     false, false, false);

        auto tr2 = makeSet({{1, 4998}, {2, 2}}, 5000);
        auto exec2 = makeExec(src, tr2);
        opts.postReruns = true;
        expectResult("subset weak novel post-replay",
                     oracleCompare(exec2, opts), true, true, false);

        // a novel value (3 at index 0) warns under the subset relation too
        auto src3 = makeSet({{1, 4998}, {2, 2}}, 5000);
        auto tr3 = makeSet({{1, 4998}, {3, 2}}, 5000);
        auto exec3 = makeExec(src3, tr3);
        exec3.sourceTotal.arity = 1;
        exec3.transformedTotal.arity = 1;
        auto r3 = oracleCompare(exec3, opts);
        expectResult("subset novel value post-replay", r3, true, true, false);
        expectMessageContains("subset novel value message",
                              r3.compare.message(), "novel value: var0=3");
    }

    // superset relation: transformed must keep every source outcome
    {
        opts.relation = OutcomeRelation::Superset;
        auto src = makeSet({{1, 2500}, {2, 2500}}, 5000);
        auto tr = makeSet({{1, 5000}}, 5000);
        auto exec = makeExec(src, tr);
        expectResult("superset common missing", oracleCompare(exec, opts),
                     false, false, false);

        auto src2 = makeSet({{1, 24999}, {2, 1}}, 25000);
        auto tr2 = makeSet({{1, 25000}}, 25000);
        auto exec2 = makeExec(src2, tr2);
        opts.postReruns = true;
        expectResult("superset rare missing not significant",
                     oracleCompare(exec2, opts), true, false, false);
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
                     oracleCompare(exec, opts), true, false, false);
    }

    std::printf("%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILURES",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
