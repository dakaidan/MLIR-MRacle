#pragma once

#include "mlir-mracle/core/types.h"
#include "mlir-mracle/execution/execution.h"

#include <cstdint>
#include <set>

namespace mlir_mracle {

// options for the oracle's comparison of joint outcome sets
struct OracleOptions {
    // the relation used to judge the joint outcome sets
    OutcomeRelation relation = OutcomeRelation::Equality;

    // the threshold percentage for judging a missing or differing state as a hard fail
    int thresholdPct = 5;

    // true if the comparison is the post-replay round judging merged data; false if the pre-replay round
    bool postReruns = false;
};

// the result of the oracle's comparison of joint outcome sets
struct OracleResult {
    CompareResult compare;

    // true if results are inconclusive and the run should be re-executed with more threads or runs
    bool needsRerun = false;
};

// Judges the joint outcome sets aggregated across the agitation sweep
OracleResult oracleCompare(const ExecutionResult &exec,
                           const OracleOptions &opts);

} // namespace mlir_mracle
