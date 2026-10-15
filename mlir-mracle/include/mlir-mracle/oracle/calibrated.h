#pragma once

#include "mlir-mracle/core/types.h"
#include "mlir-mracle/execution/execution.h"

#include <cstdint>
#include <map>
#include <vector>

namespace mlir_mracle {

// outcome counts of one pooled side, keyed by the complete joint outcome
using OutcomeCounts = std::map<JointOutcome, int64_t>;

// options of the noise-calibrated comparison
struct CalibratedOptions {
    // Largest distributional change that is not worth reporting, in squared
    // Hellinger distance. The decision bar is never lower than this, so ordinary
    // sampling noise cannot produce a verdict on its own.
    double minChange = 0.005;

    // False-alarm budget of the whole comparison, shared across the judgements
    // the campaign budget allows.
    double alphaTotal = 0.05;

    // Judgements this comparison may make before its budget is spent.
    int plannedLooks = 1;

    // Resamples that build the noise envelope.
    int permutations = 1000;

    // Seed of the resampling; a fixed seed makes a verdict reproducible.
    uint32_t seed = 0;

    // Campaign cap; the comparison asks for more campaigns while undecided.
    size_t maxCampaigns = 30;
};

enum class CalibrationVerdict { Ok, Fail, Inconclusive };

// Result of the noise-calibrated comparison of two variants' outcome
// distributions.
struct CalibrationResult {
    CompareResult compare;

    // Squared Hellinger distance between the pooled distributions, the score the
    // decision is made on.
    double score = 0.0;

    // Upper end of the distribution of scores two halves of the same campaigns
    // reach, at this budget and this false-alarm budget.
    double noiseThreshold = 0.0;

    // The practical floor the comparison refuses to decide below.
    double minChange = 0.0;

    // Bar the score is judged against: the higher of the two above.
    double decisionThreshold = 0.0;

    // Share of resamples that scored at least as high as the observed score.
    double pValue = 1.0;
    int permutations = 0;

    int64_t sourceRuns = 0;
    int64_t transformedRuns = 0;
    size_t sourceCampaigns = 0;
    size_t transformedCampaigns = 0;
    size_t sourceSupport = 0;      // distinct outcomes the source produced
    size_t transformedSupport = 0; // distinct outcomes the transformed side produced
    size_t sourceOnly = 0;         // outcomes only the source produced
    size_t transformedOnly = 0;    // outcomes only the transformed side produced

    CalibrationVerdict verdict = CalibrationVerdict::Ok;

    // True while the score sits between the practical floor and the decision bar
    // and the campaign budget could still settle it.
    bool needsRerun = false;
};

// Squared Hellinger distance between two count profiles over the union of their
// outcomes. Outcomes only one side produced contribute their own probability, so
// an emergence from nothing is measured without a pseudocount.
double squaredHellinger(const OutcomeCounts &source,
                        const OutcomeCounts &transformed);

// Judges two variants from their pooled outcome distributions. The noise
// envelope is resampled from the campaigns themselves, so compilation and
// scheduling variability at the same budget are part of the bar, and the
// comparison is symmetric: it does not use the transform relation to excuse a
// difference.
CalibrationResult calibratedCompare(const ExecutionResult &exec,
                                    const CalibratedOptions &opts);

} // namespace mlir_mracle
