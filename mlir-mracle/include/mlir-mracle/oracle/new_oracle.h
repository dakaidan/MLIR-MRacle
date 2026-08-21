#pragma once

#include "mlir-mracle/execution/execution.h"

#include <cstdint>

namespace mlir_mracle {

struct NewOracleOptions {
    // the metamorphic relation under test; the oracle only judges the
    // states the relation constrains (see newOracleCompare)
    OutcomeRelation relation = OutcomeRelation::Equality;
    // fail/warn classifier for rate shifts: a rare-in-source/common-in-
    // transformed state (source rate below this percent) fails when its
    // count also exceeds the Poisson bound of the source-predicted count,
    // and a common-missing source state (source rate at or above this
    // percent) fails when its absence is Poisson-significant. Novel states
    // are exempt from the Poisson bound: a novel transformed set at or
    // above this percent of transformed runs fails, a rarer one warns
    // post-replay, and a novel value (a transformed value never seen at
    // that position in the source) always warns regardless of its count.
    int thresholdPct = 5;
    // false while the pipeline replays rare states; the pipeline's final
    // call sets it so post-replay verdicts are judged on merged data
    bool postReruns = false;
    // reserved for a later iteration: a post-WARN replay under TSan/UBSan
    // would escalate a crash found during that replay to FAIL instead of
    // reporting the rate shift; plumbing is not wired yet
    bool sanitizeTriage = false;
};

struct NewOracleResult {
    CompareResult compare;
    // true when the pre-replay judgement found a state that needs a
    // replay round (a novel outcome set or a Poisson-significant missing
    // source state) before the final verdict; never set on the final
    // post-replay call. Novel values warn directly but still keep the
    // replay loop going through compare.warn.
    bool needsRerun = false;
};

// Judges the joint outcome sets aggregated across the agitation sweep
// (thread counts, opt levels, code layout). The single-thread probes
// must agree deterministically; anything else is a hard fail. Threaded
// verdicts follow the relation: equality judges missing, novel-set,
// novel-value and shared states; subset judges novel-set and novel-value
// states; superset judges missing states. Verdict messages name the exact
// deviation (e.g. "novel value: var0=3") instead of a generic category.
NewOracleResult newOracleCompare(const ExecutionResult &exec,
                                 const NewOracleOptions &opts);

} // namespace mlir_mracle
