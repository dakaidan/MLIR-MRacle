#pragma once

#include "mlir-mracle/core/types.h"

#include <cstdint>
#include <map>
#include <vector>

namespace mlir_mracle {

// One campaign's frequency map. The campaign is the sampling unit: campaign
// boundaries are preserved into the comparison so that the uncertainty
// estimate reflects campaign-to-campaign variability.
struct CampaignRecord {
    std::vector<JointOutcome> outcomes; // sorted unique joint outcomes
    std::vector<int64_t> counts;        // occurrences per outcome, same order
    int64_t validRuns = 0;              // executions that produced an outcome
    bool usable = true;                 // false when the campaign produced nothing
};

// Verdict of the distributional comparison; Inconclusive becomes WARN once the
// campaign budget is spent.
enum class DistributionVerdict { Ok, Bad, Inconclusive };

// Evidence state of one outcome that only a single variant produces: Ok means
// the occurrence is inside the fluke band and is ignored, Candidate means more
// campaigns are needed, Fail means the violation cleared both bars.
enum class SupportVerdict { Ok, Candidate, Fail };

// Tuning for both components of the oracle
struct DistributionOptions {
    double tvdThreshold = 0.05;  // practical tolerance, delta
    int bootstrapReplicates = 2000;
    double confidenceLevel = 0.95;
    int thresholdPct = 5;        // share of the offending side's runs a violation must reach
    uint32_t bootstrapSeed = 0;  // deterministic resampling

    // Executions one side can still reach, used to tell a candidate that more
    // campaigns could establish from one they could never establish. Zero
    // disables the projection, so every candidate counts as worth pursuing.
    int64_t maxRunsPerSide = 0;
};

struct DistributionResult {
    DistributionVerdict verdict = DistributionVerdict::Inconclusive;
    double tvd = 0.0;
    double ciLow = 0.0;
    double ciHigh = 0.0;
    size_t sourceCampaigns = 0;      // usable campaigns
    size_t transformedCampaigns = 0; // usable campaigns
    size_t excludedCampaigns = 0;    // campaigns dropped for having no valid runs
    size_t sourceSupport = 0;        // distinct outcomes observed on the source side
    size_t transformedSupport = 0;
    double sourceEffCategories = 0.0; // 1 / sum(p^2), a measure of how spread the mass is
    double transformedEffCategories = 0.0;
    // estimated share of each variant's probability mass sitting on outcomes its
    // own campaigns never produced, averaged over campaigns
    double sourceUnseenMass = 0.0;
    double transformedUnseenMass = 0.0;
    int64_t sourceRuns = 0;
    int64_t transformedRuns = 0;
};

// An outcome produced by one variant only, with the evidence collected for it
struct SupportViolation {
    JointOutcome outcome;
    bool transformedOnly = true; // true when only the transformed side produces it
    int64_t count = 0;           // occurrences on the offending side
    int64_t runs = 0;            // executions on the offending side
    double rate = 0.0;
    double bandRate = 0.0;       // fluke band this outcome was judged against
    bool significant = false;    // occurrence cleared the evidence bar
    // true when the campaign budget still reachable could establish this
    // outcome; a candidate that never could is not worth sampling for
    bool establishable = true;
    SupportVerdict verdict = SupportVerdict::Ok;
};

// Normalized empirical distribution over complete joint outcomes
using OutcomeDistribution = std::map<JointOutcome, double>;

// Equal-weight mean of the per-campaign distributions; unusable campaigns are skipped
OutcomeDistribution campaignProfile(const std::vector<CampaignRecord> &campaigns);

// Relation-masked discrepancy in [0, 1]: Equality is the total variation
// distance over the union of outcomes, Subset is the probability mass the
// transformed side places on outcomes the source never showed, Superset is the
// mass the source showed that the transformed side no longer produces.
double directionalDiscrepancy(const OutcomeDistribution &source,
                             const OutcomeDistribution &transformed,
                             OutcomeRelation relation);

// P(X >= k) for X ~ Binomial(n, p)
double binomialUpperTail(int64_t k, int64_t n, double p);

// Campaign-level comparison of the two variants' outcome distributions
DistributionResult compareDistributions(
    const std::vector<CampaignRecord> &source,
    const std::vector<CampaignRecord> &transformed, OutcomeRelation relation,
    const DistributionOptions &opts);

// Per-category evidence for outcomes that only one variant produces, in the
// direction the relation judges
std::vector<SupportViolation> checkSupport(
    const std::vector<CampaignRecord> &source,
    const std::vector<CampaignRecord> &transformed, OutcomeRelation relation,
    const DistributionOptions &opts);

// Usable campaigns of one variant
size_t usableCampaignCount(const std::vector<CampaignRecord> &campaigns);

// Total executions across the usable campaigns of one variant
int64_t totalValidRuns(const std::vector<CampaignRecord> &campaigns);

// Estimated share of one variant's probability mass on outcomes never observed
// (Good-Turing: outcomes seen exactly once, averaged over campaigns). It is the
// empirical answer to "how much could be hiding", as opposed to the sampling
// bound, which assumes nothing is.
double unseenMassRate(const std::vector<CampaignRecord> &campaigns);

// Occurrences of one outcome across a campaign set, preserving campaign boundaries
int64_t campaignCount(const std::vector<CampaignRecord> &campaigns,
                      const JointOutcome &outcome);

} // namespace mlir_mracle
