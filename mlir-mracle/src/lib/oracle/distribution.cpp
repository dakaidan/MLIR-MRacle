#include "mlir-mracle/oracle/distribution.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <utility>

namespace mlir_mracle {
namespace {

// evidence bar for a single outcome; matches the strong-evidence p-value the
// legacy judges use, so both oracles treat "clearly established" alike
constexpr double kEvidencePValue = 1e-6;

// adds one campaign's normalized counts into a profile
void accumulateProfile(OutcomeDistribution &profile, const CampaignRecord &campaign) {
    if (!campaign.usable || campaign.validRuns <= 0)
        return;

    const double runs = static_cast<double>(campaign.validRuns);
    for (size_t i = 0; i < campaign.outcomes.size(); ++i) {
        const int64_t count = i < campaign.counts.size() ? campaign.counts[i] : 0;
        if (count <= 0)
            continue;
        profile[campaign.outcomes[i]] += static_cast<double>(count) / runs;
    }
}

// profile of a bootstrap sample, drawn with replacement over whole campaigns
OutcomeDistribution sampledProfile(
    const std::vector<const CampaignRecord *> &campaigns, std::mt19937 &rng) {
    OutcomeDistribution profile;
    if (campaigns.empty())
        return profile;

    std::uniform_int_distribution<size_t> pick(0, campaigns.size() - 1);
    for (size_t i = 0; i < campaigns.size(); ++i)
        accumulateProfile(profile, *campaigns[pick(rng)]);

    const double weight = static_cast<double>(campaigns.size());
    for (auto &entry : profile)
        entry.second /= weight;
    return profile;
}

// order statistic with linear interpolation between neighbours
double percentile(const std::vector<double> &sorted, double fraction) {
    if (sorted.empty())
        return 0.0;
    if (sorted.size() == 1)
        return sorted.front();

    const double position = fraction * static_cast<double>(sorted.size() - 1);
    const size_t low = static_cast<size_t>(std::floor(position));
    const size_t high = std::min(low + 1, sorted.size() - 1);
    const double frac = position - static_cast<double>(low);
    return sorted[low] + (sorted[high] - sorted[low]) * frac;
}

// Largest rate a sample of n executions cannot tell apart from zero at the
// evidence bar: a side that never showed the outcome has only bounded its rate,
// it has not proven it is zero. This assumes nothing is hiding; unseenMassRate
// supplies the empirical correction.
double samplingResolution(int64_t n) {
    if (n <= 0)
        return 1.0;
    return -std::expm1(std::log(kEvidencePValue) / static_cast<double>(n));
}

// The fluke band: the larger of what sampling alone can explain and what the
// other side's own campaigns suggest is still unobserved. Taking the larger of
// the two keeps the band a bound rather than a point estimate, so the evidence
// bar still controls how often a real difference is missed.
double flukeBand(int64_t n, double unseenMass) {
    return std::min(1.0, std::max(samplingResolution(n), unseenMass));
}

double effectiveCategories(const OutcomeDistribution &profile) {
    double sumSquares = 0.0;
    for (const auto &entry : profile)
        sumSquares += entry.second * entry.second;
    return sumSquares > 0.0 ? 1.0 / sumSquares : 0.0;
}

// outcomes of one side only, in the direction the relation judges
std::vector<JointOutcome> exclusiveOutcomes(const OutcomeDistribution &own,
                                           const OutcomeDistribution &other) {
    std::vector<JointOutcome> exclusive;
    for (const auto &entry : own)
        if (!other.count(entry.first))
            exclusive.push_back(entry.first);
    return exclusive;
}

// True when the budget still reachable could establish this outcome. It projects
// the observed rate onto the executions each side would reach, then asks whether
// the occurrence would clear the evidence bar there. A candidate that could never
// be established is not worth spending campaigns on.
bool establishableIfSampled(const SupportViolation &violation, int64_t ownRuns,
                            int64_t otherRuns, double otherUnseenMass,
                            int64_t maxRunsPerSide) {
    if (maxRunsPerSide <= 0)
        return true; // projection disabled: every candidate is worth pursuing
    if (maxRunsPerSide <= ownRuns)
        return false; // nothing left to spend, so nothing can be established

    const int64_t remaining = maxRunsPerSide - ownRuns;
    const double projectedRate =
        static_cast<double>(violation.count + static_cast<int64_t>(
                                                   violation.rate * remaining)) /
        static_cast<double>(ownRuns + remaining);
    const int64_t projectedCount =
        violation.count + static_cast<int64_t>(violation.rate * remaining);

    if (violation.transformedOnly) {
        // the other side grows too, and with it the rate it could hide
        const double projectedBand =
            flukeBand(otherRuns + remaining, otherUnseenMass);
        return binomialUpperTail(projectedCount, ownRuns + remaining,
                                 projectedBand) < kEvidencePValue;
    }
    return std::exp(static_cast<double>(ownRuns + remaining) *
                    std::log1p(-projectedRate)) < kEvidencePValue;
}

// evidence ladder shared by both directions
SupportVerdict classifyViolation(const SupportViolation &violation,
                                 int thresholdPct, double expectedNoise) {
    if (violation.significant && violation.rate * 100.0 >= thresholdPct)
        return SupportVerdict::Fail;
    if (violation.significant)
        return SupportVerdict::Candidate;
    // a count inside the band the other side's resolution explains is a fluke
    if (static_cast<double>(violation.count) > expectedNoise)
        return SupportVerdict::Candidate;
    return SupportVerdict::Ok;
}

} // namespace

OutcomeDistribution campaignProfile(const std::vector<CampaignRecord> &campaigns) {
    OutcomeDistribution profile;
    size_t used = 0;
    for (const auto &campaign : campaigns) {
        if (!campaign.usable || campaign.validRuns <= 0)
            continue;
        accumulateProfile(profile, campaign);
        ++used;
    }

    if (used > 1)
        for (auto &entry : profile)
            entry.second /= static_cast<double>(used);
    return profile;
}

double directionalDiscrepancy(const OutcomeDistribution &source,
                             const OutcomeDistribution &transformed,
                             OutcomeRelation relation) {
    double total = 0.0;

    if (relation == OutcomeRelation::Subset) {
        for (const auto &entry : exclusiveOutcomes(transformed, source))
            total += transformed.at(entry);
        return std::min(1.0, total);
    }

    if (relation == OutcomeRelation::Superset) {
        for (const auto &entry : exclusiveOutcomes(source, transformed))
            total += source.at(entry);
        return std::min(1.0, total);
    }

    for (const auto &entry : source) {
        const auto other = transformed.find(entry.first);
        total += std::fabs(entry.second -
                           (other == transformed.end() ? 0.0 : other->second));
    }
    for (const auto &entry : transformed)
        if (!source.count(entry.first))
            total += entry.second;
    return std::min(1.0, 0.5 * total);
}

double binomialUpperTail(int64_t k, int64_t n, double p) {
    if (k <= 0)
        return 1.0;
    if (n <= 0 || k > n)
        return 0.0;
    if (p <= 0.0)
        return 0.0;
    if (p >= 1.0)
        return 1.0;

    // the lower tail is summed in log space: the terms span hundreds of orders
    // of magnitude at the sample sizes the oracle uses, so a linear sum or an
    // incomplete-beta evaluation would lose the small-count region entirely
    const double logP = std::log(p);
    const double logQ = std::log1p(-p);

    double term = static_cast<double>(n) * logQ; // pmf(0)
    double sum = term;
    for (int64_t i = 1; i < k; ++i) {
        term += std::log(static_cast<double>(n - i + 1)) -
                std::log(static_cast<double>(i)) + logP - logQ;
        const double high = std::max(sum, term);
        const double low = std::min(sum, term);
        sum = high + std::log1p(std::exp(low - high));
    }

    if (sum >= 0.0)
        return 0.0;
    return 1.0 - std::exp(sum);
}

size_t usableCampaignCount(const std::vector<CampaignRecord> &campaigns) {
    size_t usable = 0;
    for (const auto &campaign : campaigns)
        if (campaign.usable && campaign.validRuns > 0)
            ++usable;
    return usable;
}

int64_t totalValidRuns(const std::vector<CampaignRecord> &campaigns) {
    int64_t runs = 0;
    for (const auto &campaign : campaigns)
        if (campaign.usable && campaign.validRuns > 0)
            runs += campaign.validRuns;
    return runs;
}

double unseenMassRate(const std::vector<CampaignRecord> &campaigns) {
    double total = 0.0;
    size_t used = 0;

    for (const auto &campaign : campaigns) {
        if (!campaign.usable || campaign.validRuns <= 0)
            continue;
        int64_t singletons = 0;
        for (size_t i = 0; i < campaign.counts.size(); ++i)
            if (campaign.counts[i] == 1)
                ++singletons;
        total += static_cast<double>(singletons) /
                 static_cast<double>(campaign.validRuns);
        ++used;
    }

    if (used == 0)
        return 0.0;
    return std::min(1.0, total / static_cast<double>(used));
}

int64_t campaignCount(const std::vector<CampaignRecord> &campaigns,
                      const JointOutcome &outcome) {
    int64_t count = 0;
    for (const auto &campaign : campaigns) {
        if (!campaign.usable || campaign.validRuns <= 0)
            continue;
        const auto it = std::lower_bound(campaign.outcomes.begin(),
                                         campaign.outcomes.end(), outcome);
        if (it == campaign.outcomes.end() || *it != outcome)
            continue;
        const size_t index = static_cast<size_t>(
            std::distance(campaign.outcomes.begin(), it));
        if (index < campaign.counts.size())
            count += campaign.counts[index];
    }
    return count;
}

DistributionResult compareDistributions(
    const std::vector<CampaignRecord> &source,
    const std::vector<CampaignRecord> &transformed, OutcomeRelation relation,
    const DistributionOptions &opts) {
    DistributionResult result;

    std::vector<const CampaignRecord *> sourceCampaigns;
    std::vector<const CampaignRecord *> transformedCampaigns;
    for (const auto &campaign : source) {
        if (campaign.usable && campaign.validRuns > 0)
            sourceCampaigns.push_back(&campaign);
        else
            ++result.excludedCampaigns;
    }
    for (const auto &campaign : transformed) {
        if (campaign.usable && campaign.validRuns > 0)
            transformedCampaigns.push_back(&campaign);
        else
            ++result.excludedCampaigns;
    }

    result.sourceCampaigns = sourceCampaigns.size();
    result.transformedCampaigns = transformedCampaigns.size();
    result.sourceRuns = totalValidRuns(source);
    result.transformedRuns = totalValidRuns(transformed);

    const OutcomeDistribution sourceProfile = campaignProfile(source);
    const OutcomeDistribution transformedProfile = campaignProfile(transformed);
    result.sourceSupport = sourceProfile.size();
    result.transformedSupport = transformedProfile.size();
    result.sourceEffCategories = effectiveCategories(sourceProfile);
    result.transformedEffCategories = effectiveCategories(transformedProfile);
    result.sourceUnseenMass = unseenMassRate(source);
    result.transformedUnseenMass = unseenMassRate(transformed);

    result.tvd = directionalDiscrepancy(sourceProfile, transformedProfile, relation);

    // a single campaign cannot be resampled, so it cannot support a verdict
    const bool resamplable = sourceCampaigns.size() >= 2 &&
                             transformedCampaigns.size() >= 2 &&
                             opts.bootstrapReplicates > 0 &&
                             opts.confidenceLevel > 0.0 &&
                             opts.confidenceLevel < 1.0;
    if (!resamplable) {
        result.ciLow = result.tvd;
        result.ciHigh = result.tvd;
        result.verdict = DistributionVerdict::Inconclusive;
        return result;
    }

    std::mt19937 rng(opts.bootstrapSeed);
    std::vector<double> replicates;
    replicates.reserve(static_cast<size_t>(opts.bootstrapReplicates));
    for (int replicate = 0; replicate < opts.bootstrapReplicates; ++replicate) {
        const OutcomeDistribution sourceSample =
            sampledProfile(sourceCampaigns, rng);
        const OutcomeDistribution transformedSample =
            sampledProfile(transformedCampaigns, rng);
        replicates.push_back(
            directionalDiscrepancy(sourceSample, transformedSample, relation));
    }

    std::sort(replicates.begin(), replicates.end());
    const double tail = (1.0 - opts.confidenceLevel) / 2.0;
    result.ciLow = percentile(replicates, tail);
    result.ciHigh = percentile(replicates, 1.0 - tail);

    if (result.ciHigh <= opts.tvdThreshold)
        result.verdict = DistributionVerdict::Ok;
    else if (result.ciLow > opts.tvdThreshold)
        result.verdict = DistributionVerdict::Bad;
    else
        result.verdict = DistributionVerdict::Inconclusive;

    return result;
}

std::vector<SupportViolation> checkSupport(
    const std::vector<CampaignRecord> &source,
    const std::vector<CampaignRecord> &transformed, OutcomeRelation relation,
    const DistributionOptions &opts) {
    std::vector<SupportViolation> violations;

    const int64_t sourceRuns = totalValidRuns(source);
    const int64_t transformedRuns = totalValidRuns(transformed);
    if (sourceRuns <= 0 || transformedRuns <= 0)
        return violations;

    const OutcomeDistribution sourceProfile = campaignProfile(source);
    const OutcomeDistribution transformedProfile = campaignProfile(transformed);

    const bool judgeNovel = relation == OutcomeRelation::Equality ||
                            relation == OutcomeRelation::Subset;
    const bool judgeMissing = relation == OutcomeRelation::Equality ||
                              relation == OutcomeRelation::Superset;

    if (judgeNovel) {
        // The source never produced these outcomes. Whether that is sampling or
        // a real absence depends on both bounds: how much a sample this size can
        // hide, and how much the source's own singletons suggest is hiding.
        const double sourceUnseenMass = unseenMassRate(source);
        const double band = flukeBand(sourceRuns, sourceUnseenMass);
        const double expectedNoise = band * static_cast<double>(transformedRuns);
        for (const auto &outcome :
             exclusiveOutcomes(transformedProfile, sourceProfile)) {
            SupportViolation violation;
            violation.outcome = outcome;
            violation.transformedOnly = true;
            violation.count = campaignCount(transformed, outcome);
            violation.runs = transformedRuns;
            violation.rate = static_cast<double>(violation.count) /
                             static_cast<double>(transformedRuns);
            violation.bandRate = band;
            violation.significant =
                binomialUpperTail(violation.count, transformedRuns, band) <
                kEvidencePValue;
            violation.establishable = establishableIfSampled(
                violation, transformedRuns, sourceRuns, sourceUnseenMass,
                opts.maxRunsPerSide);
            violation.verdict =
                classifyViolation(violation, opts.thresholdPct, expectedNoise);
            violations.push_back(std::move(violation));
        }
    }

    if (judgeMissing) {
        // A source outcome the transformed side never produced is a question
        // about the transformed side's sampling alone: the outcome is known to
        // exist, so no unseen-mass correction applies to it.
        const double band = flukeBand(transformedRuns, 0.0);
        const double expectedNoise = band * static_cast<double>(sourceRuns);
        for (const auto &outcome :
             exclusiveOutcomes(sourceProfile, transformedProfile)) {
            SupportViolation violation;
            violation.outcome = outcome;
            violation.transformedOnly = false;
            violation.count = campaignCount(source, outcome);
            violation.runs = sourceRuns;
            violation.rate = static_cast<double>(violation.count) /
                             static_cast<double>(sourceRuns);
            violation.bandRate = band;
            // the transformed side produced none of them, which is binomial
            // evidence against a rate this high
            violation.significant =
                std::exp(static_cast<double>(transformedRuns) *
                         std::log1p(-violation.rate)) < kEvidencePValue;
            violation.establishable = establishableIfSampled(
                violation, sourceRuns, transformedRuns, /*otherUnseenMass=*/0.0,
                opts.maxRunsPerSide);
            violation.verdict =
                classifyViolation(violation, opts.thresholdPct, expectedNoise);
            violations.push_back(std::move(violation));
        }
    }

    return violations;
}

} // namespace mlir_mracle
