#pragma once

#include "mlir-mracle/core/types.h"
#include "mlir-mracle/execution/execution.h"
#include "mlir-mracle/oracle/distribution.h"

#include <cstdint>
#include <vector>

namespace mlir_mracle {

// Options for the oracle's distribution comparison and support check
struct DistributionOracleOptions {
    // The relation gates both components: Equality judges both directions,
    // Subset only the outcomes the transformed side adds, Superset only the
    // outcomes it drops.
    OutcomeRelation relation = OutcomeRelation::Equality;

    // Campaign budget; the oracle asks for more campaigns while it is inconclusive
    size_t maxCampaigns = 50;

    DistributionOptions distribution;
};

// Result of the oracle's comparison of the two variants' outcome distributions
struct DistributionOracleResult {
    CompareResult compare;

    // true if the comparison is inconclusive and more campaigns should be collected
    bool needsRerun = false;

    // Evidence behind the verdict, recorded in the run artifacts for debugging;
    // never required to read the verdict itself.
    DistributionResult distribution;
    std::vector<SupportViolation> support;
};

// Judges the per-campaign outcome distributions of the source and transformed
// programs, and the outcomes only one of them produces.
DistributionOracleResult distributionOracleCompare(const ExecutionResult &exec,
                           const DistributionOracleOptions &opts);

} // namespace mlir_mracle
