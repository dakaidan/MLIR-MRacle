// Regression tests for the distributional oracle. The decision rules are
// public API, so this links the library the pipeline runs instead of including
// a translation unit.
//
// The bootstrap is exercised only through stored campaign records, which is
// exactly the guarantee under test: resampling must never compile or execute a
// program.
#include "mlir-mracle/oracle/distribution.h"
#include "mlir-mracle/oracle/distribution_oracle.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace {

using mlir_mracle::CampaignRecord;
using mlir_mracle::CompiledBinary;
using mlir_mracle::DistributionOptions;
using mlir_mracle::DistributionResult;
using mlir_mracle::DistributionVerdict;
using mlir_mracle::ExecutionResult;
using mlir_mracle::JointOutcome;
using mlir_mracle::ObservedOutcomeSet;
using mlir_mracle::DistributionOracleOptions;
using mlir_mracle::DistributionOracleResult;
using mlir_mracle::OutcomeDistribution;
using mlir_mracle::OutcomeRelation;
using mlir_mracle::SupportVerdict;
using mlir_mracle::SupportViolation;

int failures = 0;

void report(const char *name, bool pass, const std::string &detail) {
    std::printf("[%s] %-46s %s\n", pass ? "PASS" : "FAIL", name,
                detail.c_str());
    if (!pass)
        ++failures;
}

// one campaign of single-value outcomes, given as {value, count} pairs
CampaignRecord campaign(const std::vector<std::pair<int64_t, int64_t>> &entries) {
    CampaignRecord record;
    int64_t runs = 0;
    for (const auto &[value, count] : entries) {
        record.outcomes.push_back({value});
        record.counts.push_back(count);
        runs += count;
    }
    record.validRuns = runs;
    record.usable = runs > 0;
    return record;
}

// one campaign of multi-value outcomes, given as {outcome, count} pairs
CampaignRecord campaignOf(
    const std::vector<std::pair<JointOutcome, int64_t>> &entries) {
    CampaignRecord record;
    int64_t runs = 0;
    for (const auto &[outcome, count] : entries) {
        record.outcomes.push_back(outcome);
        record.counts.push_back(count);
        runs += count;
    }
    record.validRuns = runs;
    record.usable = runs > 0;
    return record;
}

std::vector<CampaignRecord> repeat(const CampaignRecord &record, size_t times) {
    return std::vector<CampaignRecord>(times, record);
}

// the violation for one outcome, on the side that produced it
const SupportViolation *findViolation(
    const std::vector<SupportViolation> &found, const JointOutcome &outcome,
    bool transformedOnly) {
    for (const auto &violation : found)
        if (violation.outcome == outcome &&
            violation.transformedOnly == transformedOnly)
            return &violation;
    return nullptr;
}

DistributionOptions options(double threshold = 0.05) {
    DistributionOptions opts;
    opts.tvdThreshold = threshold;
    opts.bootstrapReplicates = 200;
    opts.confidenceLevel = 0.95;
    opts.bootstrapSeed = 7;
    return opts;
}

void expectDistribution(const char *name, const DistributionResult &result,
                        DistributionVerdict verdict, double tvd,
                        double tolerance = 1e-6) {
    const bool pass = result.verdict == verdict &&
                      std::fabs(result.tvd - tvd) <= tolerance;
    char detail[192];
    std::snprintf(detail, sizeof(detail),
                  "tvd=%.5f ci=[%.5f, %.5f] verdict=%d", result.tvd,
                  result.ciLow, result.ciHigh, static_cast<int>(result.verdict));
    report(name, pass, detail);
}

void expectViolation(const char *name,
                     const std::vector<SupportViolation> &found, size_t index,
                     SupportVerdict verdict, bool significant, int64_t count) {
    if (index >= found.size()) {
        report(name, false, "no violation at index " + std::to_string(index));
        return;
    }
    const SupportViolation &violation = found[index];
    const bool pass = violation.verdict == verdict &&
                      violation.significant == significant &&
                      violation.count == count;
    char detail[224];
    std::snprintf(detail, sizeof(detail),
                  "count=%lld significant=%d verdict=%d violations=%zu",
                  static_cast<long long>(violation.count),
                  violation.significant ? 1 : 0,
                  static_cast<int>(violation.verdict), found.size());
    report(name, pass, detail);
}

void expectResult(const char *name, const DistributionOracleResult &result, bool ok,
                  bool warn, bool needsRerun) {
    const bool pass = result.compare.ok() == ok && result.compare.warn() == warn &&
                      result.needsRerun == needsRerun;
    char detail[224];
    std::snprintf(detail, sizeof(detail), "ok=%d warn=%d rerun=%d issues=%zu",
                  result.compare.ok() ? 1 : 0, result.compare.warn() ? 1 : 0,
                  result.needsRerun ? 1 : 0, result.compare.issues.size());
    report(name, pass, detail);
}

// wraps campaign records into the harness result the oracle consumes; the
// oracle reads the per-campaign records only, so the aggregate sets the
// pipeline keeps for reporting are not needed here
ExecutionResult makeExec(const std::vector<CampaignRecord> &source,
                         const std::vector<CampaignRecord> &transformed) {
    ExecutionResult exec;
    for (size_t i = 0; i < source.size(); ++i) {
        CompiledBinary binary;
        binary.identity = {"source", static_cast<int>(i), -1, 0};
        binary.threadedTotal.outcomes = source[i].outcomes;
        binary.threadedTotal.counts = source[i].counts;
        binary.threadedTotal.totalRuns = source[i].validRuns;
        exec.sourceBinaries.push_back(std::move(binary));
    }
    for (size_t i = 0; i < transformed.size(); ++i) {
        CompiledBinary binary;
        binary.identity = {"transformed", static_cast<int>(i), -1, 0};
        binary.threadedTotal.outcomes = transformed[i].outcomes;
        binary.threadedTotal.counts = transformed[i].counts;
        binary.threadedTotal.totalRuns = transformed[i].validRuns;
        exec.transformedBinaries.push_back(std::move(binary));
    }
    return exec;
}

// the 1-thread probe both sides agree on
void addMatchingProbe(ExecutionResult &exec, int64_t value = 1) {
    ObservedOutcomeSet probe;
    probe.outcomes.push_back({value});
    probe.counts.push_back(32);
    probe.totalRuns = 32;
    exec.sourceSingleThreadTotal = probe;
    exec.transformedSingleThreadTotal = probe;
}

void addProbe(ExecutionResult &exec, int64_t sourceValue, int64_t transformedValue) {
    ObservedOutcomeSet source;
    source.outcomes.push_back({sourceValue});
    source.counts.push_back(32);
    source.totalRuns = 32;
    ObservedOutcomeSet transformed;
    transformed.outcomes.push_back({transformedValue});
    transformed.counts.push_back(32);
    transformed.totalRuns = 32;
    exec.sourceSingleThreadTotal = source;
    exec.transformedSingleThreadTotal = transformed;
}

DistributionOracleOptions oracleOptions(size_t maxCampaigns = 10, double threshold = 0.05) {
    DistributionOracleOptions opts;
    opts.maxCampaigns = maxCampaigns;
    opts.distribution = options(threshold);
    return opts;
}

// ---------------------------------------------------------------------------
// discrepancy and bootstrap
// ---------------------------------------------------------------------------

void testDiscrepancy() {
    // identical distributions: no difference at all
    {
        auto same = repeat(campaign({{1, 1000}}), 10);
        auto result = mlir_mracle::compareDistributions(
            same, same, OutcomeRelation::Equality, options());
        expectDistribution("identical campaigns", result,
                           DistributionVerdict::Ok, 0.0);
    }

    // disjoint supports: every outcome is one-sided, so the distance is 1
    {
        auto result = mlir_mracle::compareDistributions(
            repeat(campaign({{1, 1000}}), 10),
            repeat(campaign({{2, 1000}}), 10), OutcomeRelation::Equality,
            options());
        expectDistribution("disjoint supports", result,
                           DistributionVerdict::Fail, 1.0);
    }

    // hand-computed: P = [A: 1.0], Q = [A: 0.9, B: 0.1] -> tvd = 0.1
    {
        auto source = repeat(campaign({{1, 1000}}), 10);
        auto transformed = repeat(campaign({{1, 900}, {2, 100}}), 10);

        expectDistribution("hand-computed shift vs delta=0.05",
                           mlir_mracle::compareDistributions(
                               source, transformed, OutcomeRelation::Equality,
                               options(0.05)),
                           DistributionVerdict::Fail, 0.1);
        expectDistribution("hand-computed shift vs delta=0.15",
                           mlir_mracle::compareDistributions(
                               source, transformed, OutcomeRelation::Equality,
                               options(0.15)),
                           DistributionVerdict::Ok, 0.1);
        expectDistribution("interval touching delta passes",
                           mlir_mracle::compareDistributions(
                               source, transformed, OutcomeRelation::Equality,
                               options(0.1)),
                           DistributionVerdict::Ok, 0.1);
    }

    // campaign variability: campaigns that disagree widen the interval, so the
    // verdict follows the interval rather than the pooled estimate
    {
        std::vector<CampaignRecord> transformed = repeat(campaign({{1, 1000}}), 5);
        for (int i = 0; i < 5; ++i)
            transformed.push_back(campaign({{2, 1000}}));
        auto source = repeat(campaign({{1, 1000}}), 10);

        auto result = mlir_mracle::compareDistributions(
            source, transformed, OutcomeRelation::Equality, options(0.05));
        const bool pass = std::fabs(result.tvd - 0.5) < 1e-6 &&
                          result.ciHigh - result.ciLow > 0.4 &&
                          result.ciLow > 0.05 &&
                          result.verdict == DistributionVerdict::Fail;
        char detail[192];
        std::snprintf(detail, sizeof(detail),
                      "tvd=%.4f ci=[%.4f, %.4f] verdict=%d", result.tvd,
                      result.ciLow, result.ciHigh,
                      static_cast<int>(result.verdict));
        report("campaign variability widens interval", pass, detail);
    }

    // the same sampling noise around the tolerance makes a verdict impossible
    {
        std::vector<CampaignRecord> transformed;
        for (int i = 0; i < 5; ++i)
            transformed.push_back(campaign({{1, 900}, {2, 100}}));
        for (int i = 0; i < 5; ++i)
            transformed.push_back(campaign({{1, 1000}}));
        auto source = repeat(campaign({{1, 1000}}), 10);

        auto result = mlir_mracle::compareDistributions(
            source, transformed, OutcomeRelation::Equality, options(0.05));
        const bool pass = result.ciLow <= 0.05 && result.ciHigh > 0.05 &&
                          result.verdict == DistributionVerdict::Inconclusive;
        char detail[192];
        std::snprintf(detail, sizeof(detail),
                      "tvd=%.4f ci=[%.4f, %.4f] verdict=%d", result.tvd,
                      result.ciLow, result.ciHigh,
                      static_cast<int>(result.verdict));
        report("noise around delta is inconclusive", pass, detail);
    }

    // an outcome seen in a single campaign shifts the mean by 1/n only
    {
        std::vector<CampaignRecord> transformed;
        for (int i = 0; i < 10; ++i)
            transformed.push_back(i == 0 ? campaign({{1, 999}, {2, 1}})
                                         : campaign({{1, 1000}}));
        auto result = mlir_mracle::compareDistributions(
            repeat(campaign({{1, 1000}}), 10), transformed,
            OutcomeRelation::Equality, options(0.05));
        expectDistribution("single-campaign outlier is not a fail", result,
                           DistributionVerdict::Ok, 0.0001);
    }

    // equal weighting: a large campaign must not dominate a small one
    {
        std::vector<CampaignRecord> source = {campaign({{1, 1000}}),
                                              campaign({{1, 10}})};
        OutcomeDistribution profile = mlir_mracle::campaignProfile(source);
        const bool pass = profile.size() == 1 &&
                          std::fabs(profile.at({1}) - 1.0) < 1e-12;
        report("profiles are equal-weight averages", pass,
               "categories=" + std::to_string(profile.size()));

        std::vector<CampaignRecord> transformed = {campaign({{2, 1000}}),
                                                   campaign({{2, 10}})};
        expectDistribution("unequal campaign sizes stay decisive",
                           mlir_mracle::compareDistributions(
                               source, transformed, OutcomeRelation::Equality,
                               options(0.05)),
                           DistributionVerdict::Fail, 1.0);
    }

    // an unusable campaign is dropped and counted, not averaged in
    {
        std::vector<CampaignRecord> transformed = repeat(campaign({{1, 1000}}), 10);
        CampaignRecord broken;
        broken.usable = false;
        transformed.push_back(broken);

        auto result = mlir_mracle::compareDistributions(
            repeat(campaign({{1, 1000}}), 10), transformed,
            OutcomeRelation::Equality, options());
        const bool pass = result.excludedCampaigns == 1 &&
                          result.transformedCampaigns == 10 &&
                          result.verdict == DistributionVerdict::Ok;
        report("unusable campaign excluded", pass,
               "excluded=" + std::to_string(result.excludedCampaigns));
    }

    // a single campaign per side cannot be resampled, so no verdict is possible
    {
        auto one = repeat(campaign({{1, 1000}}), 1);
        expectDistribution("one campaign is inconclusive",
                           mlir_mracle::compareDistributions(
                               one, one, OutcomeRelation::Equality, options()),
                           DistributionVerdict::Inconclusive, 0.0);
    }

    // the bootstrap is reproducible for a fixed seed
    {
        auto source = repeat(campaign({{1, 900}, {2, 100}}), 10);
        auto transformed = repeat(campaign({{1, 880}, {2, 120}}), 10);
        auto first = mlir_mracle::compareDistributions(
            source, transformed, OutcomeRelation::Equality, options());
        auto second = mlir_mracle::compareDistributions(
            source, transformed, OutcomeRelation::Equality, options());
        const bool pass = first.ciLow == second.ciLow &&
                          first.ciHigh == second.ciHigh &&
                          first.tvd == second.tvd;
        char detail[160];
        std::snprintf(detail, sizeof(detail), "ci=[%.5f, %.5f]", first.ciLow,
                      first.ciHigh);
        report("bootstrap is deterministic", pass, detail);
    }

    // sparse support: hundreds of rare outcomes compare as ordinary categories
    {
        std::vector<std::pair<int64_t, int64_t>> counts;
        for (int i = 0; i < 300; ++i)
            counts.push_back({i, 3});
        counts.push_back({9000, 100});
        auto same = repeat(campaign(counts), 10);
        expectDistribution("sparse support compares as categories",
                           mlir_mracle::compareDistributions(
                               same, same, OutcomeRelation::Equality, options()),
                           DistributionVerdict::Ok, 0.0);
    }
}

// ---------------------------------------------------------------------------
// relation masking
// ---------------------------------------------------------------------------

void testRelationMasking() {
    std::vector<CampaignRecord> source = repeat(campaign({{1, 500}, {2, 500}}), 10);
    std::vector<CampaignRecord> transformed = repeat(campaign({{1, 1000}}), 10);

    expectDistribution("equality judges dropped mass",
                       mlir_mracle::compareDistributions(
                           source, transformed, OutcomeRelation::Equality,
                           options()),
                       DistributionVerdict::Fail, 0.5);
    expectDistribution("subset ignores dropped mass",
                       mlir_mracle::compareDistributions(
                           source, transformed, OutcomeRelation::Subset,
                           options()),
                       DistributionVerdict::Ok, 0.0);
    expectDistribution("superset judges dropped mass",
                       mlir_mracle::compareDistributions(
                           source, transformed, OutcomeRelation::Superset,
                           options()),
                       DistributionVerdict::Fail, 0.5);

    // the transformed side adds an outcome the source never produced
    expectDistribution("subset judges added mass",
                       mlir_mracle::compareDistributions(
                           transformed, source, OutcomeRelation::Subset,
                           options()),
                       DistributionVerdict::Fail, 0.5);
    expectDistribution("superset ignores added mass",
                       mlir_mracle::compareDistributions(
                           transformed, source, OutcomeRelation::Superset,
                           options()),
                       DistributionVerdict::Ok, 0.0);
}

// ---------------------------------------------------------------------------
// support check
// ---------------------------------------------------------------------------

void testSupportCheck() {
    // one occurrence on one side and none on the other is inside both sides'
    // bands, so there is nothing to judge and nothing is recorded
    {
        std::vector<CampaignRecord> source;
        for (int i = 0; i < 10; ++i)
            source.push_back(i == 0 ? campaign({{1, 999}, {2, 1}})
                                    : campaign({{1, 1000}}));
        auto found = mlir_mracle::checkSupport(
            source, repeat(campaign({{1, 1000}}), 10),
            OutcomeRelation::Equality, options());
        report("a lone occurrence is not judged", found.empty(),
               "violations=" + std::to_string(found.size()));
    }

    // a novel outcome above the fluke band but below the evidence bar is a
    // candidate: 20 occurrences against the source's own resolution of 13.8
    {
        std::vector<CampaignRecord> transformed;
        for (int i = 0; i < 10; ++i)
            transformed.push_back(i == 0 ? campaign({{1, 980}, {2, 20}})
                                         : campaign({{1, 1000}}));
        auto found = mlir_mracle::checkSupport(
            repeat(campaign({{1, 1000}}), 10), transformed,
            OutcomeRelation::Equality, options());
        if (found.size() != 1)
            report("novel occurrence above the fluke band", false,
                   "violations=" + std::to_string(found.size()));
        else
            expectViolation("novel occurrence above the fluke band", found, 0,
                            SupportVerdict::Candidate, false, 20);
    }

    // a missing outcome whose absence is decisive fails at any rate
    {
        std::vector<CampaignRecord> source;
        for (int i = 0; i < 10; ++i)
            source.push_back(i < 2 ? campaign({{1, 990}, {2, 10}})
                                   : campaign({{1, 1000}}));
        auto found = mlir_mracle::checkSupport(
            source, repeat(campaign({{1, 1000}}), 10),
            OutcomeRelation::Equality, options());
        if (found.size() != 1)
            report("decisive absence is a failure", false,
                   "violations=" + std::to_string(found.size()));
        else
            expectViolation("decisive absence is a failure", found, 0,
                            SupportVerdict::Fail, true, 20);
    }

    // established evidence fails however rare the outcome is: 100 of
    // 20,000 is 0.5%, well past the ~36 occurrence evidence bar
    {
        auto source = repeat(campaign({{1, 990}, {2, 10}}), 10);
        auto found = mlir_mracle::checkSupport(
            source, repeat(campaign({{1, 1000}}), 10),
            OutcomeRelation::Equality, options());
        if (found.size() != 1)
            report("established rarity fails", false,
                   "violations=" + std::to_string(found.size()));
        else
            expectViolation("established rarity fails", found, 0,
                            SupportVerdict::Fail, true, 100);
    }

    // unlikely but not established: 25 of 20,000 is past the 0.05 level and
    // short of the 1e-6 evidence bar, so it warns
    {
        std::vector<CampaignRecord> transformed;
        for (int i = 0; i < 10; ++i)
            transformed.push_back(i == 0 ? campaign({{1, 1975}, {2, 25}})
                                         : campaign({{1, 2000}}));
        auto found = mlir_mracle::checkSupport(
            repeat(campaign({{1, 2000}}), 10), transformed,
            OutcomeRelation::Equality, options());
        if (found.size() != 1)
            report("unlikely occurrence warns", false,
                   "violations=" + std::to_string(found.size()));
        else
            expectViolation("unlikely occurrence warns", found, 0,
                            SupportVerdict::Warn, false, 25);
    }

    // and it reaches the verdict as a warning rather than passing silently
    {
        std::vector<CampaignRecord> transformed;
        for (int i = 0; i < 10; ++i)
            transformed.push_back(i == 0 ? campaign({{1, 1975}, {2, 25}})
                                         : campaign({{1, 2000}}));
        auto exec = makeExec(repeat(campaign({{1, 2000}}), 10), transformed);
        addMatchingProbe(exec);
        auto result = mlir_mracle::distributionOracleCompare(exec, oracleOptions());
        expectResult("unlikely occurrence warns the run", result, true, true,
                     false);
    }

    // an outcome common enough to be unambiguous
    {
        auto source = repeat(campaign({{1, 940}, {2, 60}}), 10);
        auto found = mlir_mracle::checkSupport(
            source, repeat(campaign({{1, 1000}}), 10),
            OutcomeRelation::Equality, options());
        if (found.size() != 1)
            report("violation above the threshold fails", false,
                   "violations=" + std::to_string(found.size()));
        else
            expectViolation("violation above the threshold fails", found, 0,
                            SupportVerdict::Fail, true, 600);
    }

    // the evidence bar: 13 occurrences are inside the fluke band, 36 are not
    {
        auto withCount = [](int64_t count) {
            std::vector<CampaignRecord> transformed;
            for (int i = 0; i < 10; ++i)
                transformed.push_back(
                    i == 0 ? campaign({{1, 1000 - count}, {2, count}})
                           : campaign({{1, 1000}}));
            return transformed;
        };
        auto source = repeat(campaign({{1, 1000}}), 10);

        auto below = mlir_mracle::checkSupport(source, withCount(13),
                                              OutcomeRelation::Equality,
                                              options());
        auto middle = mlir_mracle::checkSupport(source, withCount(20),
                                               OutcomeRelation::Equality,
                                               options());
        auto above = mlir_mracle::checkSupport(source, withCount(36),
                                              OutcomeRelation::Equality,
                                              options());
        const bool belowOk = below.empty();
        const bool middleOk = middle.size() == 1 &&
                              middle[0].verdict == SupportVerdict::Candidate;
        const bool aboveOk = above.size() == 1 &&
                             above[0].verdict == SupportVerdict::Fail;
        report("the ladder is 13 ignored, 20 candidate, 36 established",
               belowOk && middleOk && aboveOk,
               "13=" + std::to_string(below.size()) + " 20=" +
                   std::to_string(middle.size()) + " 36=" +
                   std::to_string(above.size()));
    }

    // sub-type relation judges one direction only
    {
        auto source = repeat(campaign({{1, 990}, {2, 10}}), 10);
        auto transformed = repeat(campaign({{1, 1000}}), 10);

        auto subset = mlir_mracle::checkSupport(source, transformed,
                                                OutcomeRelation::Subset,
                                                options());
        report("subset ignores a missing outcome", subset.empty(),
               "violations=" + std::to_string(subset.size()));

        auto superset = mlir_mracle::checkSupport(source, transformed,
                                                  OutcomeRelation::Superset,
                                                  options());
        if (superset.size() != 1)
            report("superset judges a missing outcome", false,
                   "violations=" + std::to_string(superset.size()));
        else {
            expectViolation("superset missing outcome evidence", superset, 0,
                            SupportVerdict::Fail, true, 100);
            report("superset marks the offending side",
                   !superset[0].transformedOnly, "side=source");
        }
    }

    // a source outcome at a high rate that the transformed side never
    // produced is a fail
    {
        auto found = mlir_mracle::checkSupport(
            repeat(campaign({{1, 900}, {2, 100}}), 10),
            repeat(campaign({{1, 1000}}), 10), OutcomeRelation::Superset,
            options());
        if (found.size() != 1)
            report("common missing outcome fails", false,
                   "violations=" + std::to_string(found.size()));
        else
            expectViolation("common missing outcome fails", found, 0,
                            SupportVerdict::Fail, true, 1000);
    }
}

// ---------------------------------------------------------------------------
// estimated fluke band and the stopping rule
// ---------------------------------------------------------------------------

void testUnseenMassBand() {
    // a source whose campaigns are full of one-off outcomes may still be hiding
    // mass, so the band widens to match its own singletons
    {
        std::vector<std::pair<int64_t, int64_t>> bursty;
        for (int i = 0; i < 100; ++i)
            bursty.push_back({i, 1});       // 100 outcomes seen exactly once
        bursty.push_back({9000, 1900});     // plus one dominant outcome
        std::vector<CampaignRecord> source =
            repeat(campaign(bursty), 10);   // 100 singletons per 2000 runs

        const double unseen = mlir_mracle::unseenMassRate(source);
        report("unseen mass estimated from singletons",
               std::fabs(unseen - 0.05) < 1e-9,
               "unseen=" + std::to_string(unseen));

        // 0.5% of runs: within what a source hiding 5% of its mass could explain,
        // so it is left alone rather than called a violation
        std::vector<CampaignRecord> small;
        for (int i = 0; i < 10; ++i)
            small.push_back(i == 0 ? campaign({{8000, 1900}, {8001, 100}})
                                   : campaign({{8000, 2000}}));
        auto foundSmall = mlir_mracle::checkSupport(source, small,
                                                   OutcomeRelation::Equality,
                                                   options());
        const SupportViolation *smallNovel = findViolation(foundSmall, {8001}, true);
        if (smallNovel == nullptr) {
            report("bursty source excuses a small novel outcome", false,
                   "violations=" + std::to_string(foundSmall.size()));
        } else {
            report("bursty source excuses a small novel outcome",
                   std::fabs(smallNovel->bandRate - 0.05) < 1e-9 &&
                       smallNovel->verdict == SupportVerdict::Ok,
                   "band=" + std::to_string(smallNovel->bandRate) +
                       " verdict=" + std::to_string(static_cast<int>(smallNovel->verdict)));
        }

        // 6% of runs clears the band and the detection threshold, so the widened
        // band does not excuse a real difference
        std::vector<CampaignRecord> large;
        for (int i = 0; i < 10; ++i)
            large.push_back(i == 0 ? campaign({{8000, 800}, {8001, 1200}})
                                   : campaign({{8000, 2000}}));
        auto foundLarge = mlir_mracle::checkSupport(source, large,
                                                   OutcomeRelation::Equality,
                                                   options());
        const SupportViolation *largeNovel = findViolation(foundLarge, {8001}, true);
        if (largeNovel == nullptr) {
            report("wide band still fails a large difference", false,
                   "violations=" + std::to_string(foundLarge.size()));
        } else {
            report("wide band still fails a large difference",
                   largeNovel->significant &&
                       largeNovel->verdict == SupportVerdict::Fail,
                   "rate=" + std::to_string(largeNovel->rate) +
                       " significant=" + std::to_string(largeNovel->significant));
        }
    }

    // a source with no one-off outcomes keeps the sampling bound, so the same
    // 0.5% occurrence is established instead of excused
    {
        std::vector<CampaignRecord> source = repeat(campaign({{9000, 2000}}), 10);
        std::vector<CampaignRecord> transformed;
        for (int i = 0; i < 10; ++i)
            transformed.push_back(i == 0 ? campaign({{8000, 1900}, {8001, 100}})
                                         : campaign({{8000, 2000}}));
        auto found = mlir_mracle::checkSupport(source, transformed,
                                              OutcomeRelation::Equality,
                                              options());
        const SupportViolation *novel = findViolation(found, {8001}, true);
        if (novel == nullptr) {
            report("band stays at the sampling bound", false,
                   "novel outcome not reported, violations=" +
                       std::to_string(found.size()));
        } else {
            const bool pass = novel->bandRate < 0.001 && novel->significant &&
                              novel->verdict == SupportVerdict::Fail;
            report("band stays at the sampling bound", pass,
                   "band=" + std::to_string(novel->bandRate) +
                       " significant=" + std::to_string(novel->significant));
        }
    }
}

void testStoppingRule() {
    // a candidate at 0.1% can be established once the budget grows
    {
        std::vector<CampaignRecord> transformed;
        for (int i = 0; i < 10; ++i)
            transformed.push_back(i == 0 ? campaign({{1, 1980}, {2, 20}})
                                         : campaign({{1, 2000}}));
        auto withRoom = options();
        withRoom.maxRunsPerSide = 100000;
        auto found = mlir_mracle::checkSupport(repeat(campaign({{1, 2000}}), 10),
                                              transformed,
                                              OutcomeRelation::Equality, withRoom);
        const bool pass = found.size() == 1 && found[0].establishable;
        report("candidate is worth sampling for", pass,
               found.empty()
                   ? "no violation"
                   : "rate=" + std::to_string(found[0].rate) +
                         " establishable=" + std::to_string(found[0].establishable));
    }

    // the same candidate against a budget with nothing left in it is a dead end
    {
        std::vector<CampaignRecord> transformed;
        for (int i = 0; i < 10; ++i)
            transformed.push_back(i == 0 ? campaign({{1, 1980}, {2, 20}})
                                         : campaign({{1, 2000}}));
        auto noRoom = options();
        noRoom.maxRunsPerSide = 20000; // exactly what has already been spent
        auto found = mlir_mracle::checkSupport(repeat(campaign({{1, 2000}}), 10),
                                              transformed,
                                              OutcomeRelation::Equality, noRoom);
        const SupportViolation *novel = findViolation(found, {2}, true);
        if (novel == nullptr) {
            report("hopeless candidate is a dead end", false,
                   "violations=" + std::to_string(found.size()));
        } else {
            report("hopeless candidate is a dead end", !novel->establishable,
                   "rate=" + std::to_string(novel->rate) + " establishable=" +
                       std::to_string(novel->establishable));
        }
    }

    // with the budget spent and only a dead end left, the run is ok and no
    // further sampling is requested
    {
        std::vector<CampaignRecord> transformed;
        for (int i = 0; i < 10; ++i)
            transformed.push_back(i == 0 ? campaign({{1, 1980}, {2, 20}})
                                         : campaign({{1, 2000}}));
        auto exec = makeExec(repeat(campaign({{1, 2000}}), 10), transformed);
        addMatchingProbe(exec);

        auto opts = oracleOptions(50);
        opts.distribution.maxRunsPerSide = 20000; // nothing left to spend
        expectResult("dead end candidate is ok",
                     mlir_mracle::distributionOracleCompare(exec, opts), true, false, false);
    }

    // a live candidate at the cap is recorded without failing the run
    {
        std::vector<CampaignRecord> transformed;
        for (int i = 0; i < 10; ++i)
            transformed.push_back(i == 0 ? campaign({{1, 1900}, {2, 100}})
                                         : campaign({{1, 2000}}));
        auto exec = makeExec(repeat(campaign({{1, 2000}}), 10), transformed);
        addMatchingProbe(exec);

        auto opts = oracleOptions(10);
        opts.distribution.maxRunsPerSide = 100000;
        expectResult("established candidate at the cap fails",
                     mlir_mracle::distributionOracleCompare(exec, opts), false, false,
                     false);
    }
}

// ---------------------------------------------------------------------------
// binomial tail
// ---------------------------------------------------------------------------

void testBinomialTail() {
    report("binomial tail bounds",
           mlir_mracle::binomialUpperTail(0, 100, 0.5) == 1.0 &&
               mlir_mracle::binomialUpperTail(101, 100, 0.5) == 0.0,
           "P(X>=0)=1 and P(X>n)=0");

    // P(X >= 1) = 1 - (1 - p)^n
    {
        const double expected = 1.0 - std::pow(1.0 - 0.01, 200.0);
        const double actual = mlir_mracle::binomialUpperTail(1, 200, 0.01);
        char detail[160];
        std::snprintf(detail, sizeof(detail), "expected=%.12f actual=%.12f",
                      expected, actual);
        report("binomial tail matches closed form",
               std::fabs(actual - expected) < 1e-12, detail);
    }
}

// ---------------------------------------------------------------------------
// oracle verdicts and campaign budget
// ---------------------------------------------------------------------------

void testOracleVerdicts() {
    // identical runs are ok and carry their evidence
    {
        auto same = repeat(campaign({{1, 1000}}), 10);
        auto exec = makeExec(same, same);
        addMatchingProbe(exec);
        auto result = mlir_mracle::distributionOracleCompare(exec, oracleOptions());
        expectResult("matching variants are ok", result, true, false, false);
        report("ok run records its evidence",
               result.distribution.verdict == DistributionVerdict::Ok &&
                   result.distribution.sourceCampaigns == 10,
               "campaigns=" +
                   std::to_string(result.distribution.sourceCampaigns));
    }

    // a distributional shift is a hard fail
    {
        auto exec = makeExec(repeat(campaign({{1, 1000}}), 10),
                             repeat(campaign({{2, 1000}}), 10));
        addMatchingProbe(exec);
        expectResult("distributional shift fails",
                     mlir_mracle::distributionOracleCompare(exec, oracleOptions()), false,
                     false, false);
    }

    // a rate shift between outcomes both sides produce trips no support check,
    // so the discrepancy alone has to carry the verdict
    {
        auto exec = makeExec(repeat(campaign({{1, 500}, {2, 500}}), 10),
                             repeat(campaign({{1, 900}, {2, 100}}), 10));
        addMatchingProbe(exec);
        auto result = mlir_mracle::distributionOracleCompare(exec, oracleOptions());
        expectResult("rate shift on shared outcomes fails", result, false, false,
                     false);
        report("shared outcomes raise no support violation",
               result.support.empty(),
               "violations=" + std::to_string(result.support.size()));
    }

    // a support violation fails while the discrepancy stays under tolerance:
    // the transformed side drops an outcome the source produced in 2% of runs,
    // which is only 2% of probability mass against a 5% tolerance
    {
        auto exec = makeExec(repeat(campaign({{1, 980}, {2, 20}}), 10),
                             repeat(campaign({{1, 1000}}), 10));
        addMatchingProbe(exec);
        auto result = mlir_mracle::distributionOracleCompare(exec, oracleOptions(10, 0.05));
        expectResult("support violation fails under tolerance", result, false,
                     false, false);
        report("discrepancy stayed under tolerance",
               result.distribution.tvd < 0.05,
               "tvd=" + std::to_string(result.distribution.tvd));
        bool named = false;
        std::string text;
        for (const auto &issue : result.compare.issues)
            if (issue.outcome == "[2]" &&
                issue.severity == mlir_mracle::IssueSeverity::Fail) {
                named = issue.reason == "missing outcome [2] not found in "
                                        "transformed";
                text = issue.reason;
            }
        report("support violation names the outcome", named,
               text.empty() ? "issues=" +
                                  std::to_string(result.compare.issues.size())
                            : text);
    }

    // an unresolved candidate buys more campaigns while the budget allows
    {
        std::vector<CampaignRecord> transformed;
        for (int i = 0; i < 10; ++i)
            transformed.push_back(i == 0 ? campaign({{1, 980}, {2, 20}})
                                         : campaign({{1, 1000}}));
        auto exec = makeExec(repeat(campaign({{1, 1000}}), 10), transformed);
        addMatchingProbe(exec);

        expectResult("candidate buys campaigns",
                     mlir_mracle::distributionOracleCompare(exec, oracleOptions(12)), true,
                     false, true);

        // a candidate too rare to confirm is not worth a verdict of its own: it
        // stays recorded as evidence and the run is ok
        auto capped = mlir_mracle::distributionOracleCompare(exec, oracleOptions(10));
        expectResult("candidate at the cap is ok", capped, true, false, false);
        report("rare candidate stays recorded",
               capped.support.size() == 1 &&
                   capped.support[0].verdict == SupportVerdict::Candidate,
               "violations=" + std::to_string(capped.support.size()));
    }

    // the 1-thread probe is judged independently of the distribution
    {
        auto same = repeat(campaign({{1, 1000}}), 10);
        auto exec = makeExec(same, same);
        addProbe(exec, 1, 9);
        auto result = mlir_mracle::distributionOracleCompare(exec, oracleOptions());
        bool matched = false;
        for (const auto &issue : result.compare.issues)
            if (issue.reason.rfind("single-thread results differ", 0) == 0)
                matched = true;
        report("1-thread mismatch fails", matched && !result.compare.ok(),
               "issues=" + std::to_string(result.compare.issues.size()));
    }

    // an arity change lands on disjoint supports, so it fails on the
    // distribution alone
    {
        auto exec = makeExec(repeat(campaignOf({{{1, 2}, 1000}}), 10),
                             repeat(campaignOf({{{1, 2, 3}, 1000}}), 10));
        addMatchingProbe(exec);
        expectResult("arity change fails",
                     mlir_mracle::distributionOracleCompare(exec, oracleOptions()), false,
                     false, false);
    }

    // 300 of 20,000 (1.5%) is a categorical change and fails, in every relation
    // that judges the direction it appears in
    {
        auto novelTransformed = [&] {
            std::vector<CampaignRecord> transformed;
            for (int i = 0; i < 10; ++i)
                transformed.push_back(i == 0 ? campaign({{1, 1700}, {2, 300}})
                                             : campaign({{1, 2000}}));
            return transformed;
        };

        auto added = makeExec(repeat(campaign({{1, 2000}}), 10), novelTransformed());
        addMatchingProbe(added);
        for (const auto &entry :
             std::vector<std::pair<const char *, OutcomeRelation>>{
                 {"equality", OutcomeRelation::Equality},
                 {"subset", OutcomeRelation::Subset}}) {
            auto opts = oracleOptions();
            opts.relation = entry.second;
            expectResult((std::string("300 of 20,000 fails under ") +
                          entry.first).c_str(),
                         mlir_mracle::distributionOracleCompare(added, opts), false, false,
                         false);
        }

        auto supersetAdded = oracleOptions();
        supersetAdded.relation = OutcomeRelation::Superset;
        expectResult("300 added is not superset's business",
                     mlir_mracle::distributionOracleCompare(added, supersetAdded), true,
                     false, false);

        // and the mirror: a dropped outcome
        auto dropped = makeExec(novelTransformed(), repeat(campaign({{1, 2000}}), 10));
        addMatchingProbe(dropped);
        for (const auto &entry :
             std::vector<std::pair<const char *, OutcomeRelation>>{
                 {"equality", OutcomeRelation::Equality},
                 {"superset", OutcomeRelation::Superset}}) {
            auto opts = oracleOptions();
            opts.relation = entry.second;
            expectResult((std::string("300 dropped fails under ") +
                          entry.first).c_str(),
                         mlir_mracle::distributionOracleCompare(dropped, opts), false, false,
                         false);
        }

        auto subsetDropped = oracleOptions();
        subsetDropped.relation = OutcomeRelation::Subset;
        expectResult("300 dropped is not subset's business",
                     mlir_mracle::distributionOracleCompare(dropped, subsetDropped), true,
                     false, false);
    }

    // 1 -> 300: the quiet side's single occurrence is inside its own band, so the
    // outcome is not meaningfully present there and the jump is categorical
    {
        auto sparseSource = [&] {
            std::vector<CampaignRecord> source;
            for (int i = 0; i < 10; ++i)
                source.push_back(i == 0 ? campaign({{1, 1999}, {2, 1}})
                                        : campaign({{1, 2000}}));
            return source;
        };
        auto dense = [&] {
            std::vector<CampaignRecord> transformed;
            for (int i = 0; i < 10; ++i)
                transformed.push_back(i == 0 ? campaign({{1, 1700}, {2, 300}})
                                             : campaign({{1, 2000}}));
            return transformed;
        };

        auto added = makeExec(sparseSource(), dense());
        addMatchingProbe(added);
        for (const auto &entry :
             std::vector<std::pair<const char *, OutcomeRelation>>{
                 {"equality", OutcomeRelation::Equality},
                 {"subset", OutcomeRelation::Subset}}) {
            auto opts = oracleOptions();
            opts.relation = entry.second;
            expectResult((std::string("1 -> 300 fails under ") + entry.first).c_str(),
                         mlir_mracle::distributionOracleCompare(added, opts), false, false,
                         false);
        }

        auto dropped = makeExec(dense(), sparseSource());
        addMatchingProbe(dropped);
        for (const auto &entry :
             std::vector<std::pair<const char *, OutcomeRelation>>{
                 {"equality", OutcomeRelation::Equality},
                 {"superset", OutcomeRelation::Superset}}) {
            auto opts = oracleOptions();
            opts.relation = entry.second;
            expectResult((std::string("300 -> 1 fails under ") + entry.first).c_str(),
                         mlir_mracle::distributionOracleCompare(dropped, opts), false, false,
                         false);
        }

        // and the boundary: 1 -> 3 is inside both bands, so nothing is claimed
        auto tiny = [&] {
            std::vector<CampaignRecord> transformed;
            for (int i = 0; i < 10; ++i)
                transformed.push_back(i == 0 ? campaign({{1, 1997}, {2, 3}})
                                             : campaign({{1, 2000}}));
            return transformed;
        };
        auto small = makeExec(sparseSource(), tiny());
        addMatchingProbe(small);
        expectResult("1 -> 3 is not claimed", mlir_mracle::distributionOracleCompare(small, oracleOptions()),
                     true, false, false);
    }

    // 150 -> 999: both sides produce the outcome, so support says nothing and
    // only the magnitude of the shift is left. Evenly spread across campaigns it
    // is 4.24 percentage points of mass, just inside the tolerance
    {
        std::vector<CampaignRecord> source;
        for (int i = 0; i < 10; ++i)
            source.push_back(campaign({{1, 1985}, {2, 15}}));
        std::vector<CampaignRecord> spread;
        for (int i = 0; i < 9; ++i)
            spread.push_back(campaign({{1, 1900}, {2, 100}}));
        spread.push_back(campaign({{1, 1901}, {2, 99}}));

        auto exec = makeExec(source, spread);
        addMatchingProbe(exec);
        auto result = mlir_mracle::distributionOracleCompare(exec, oracleOptions());
        char detail[192];
        std::snprintf(detail, sizeof(detail),
                      "tvd=%.5f ci=[%.5f, %.5f] violations=%zu verdict=%s",
                      result.distribution.tvd, result.distribution.ciLow,
                      result.distribution.ciHigh, result.support.size(),
                      result.compare.ok() ? (result.compare.warn() ? "WARN" : "OK")
                                          : "FAIL");
        report("150 -> 999 spread evenly is tolerated",
               result.support.empty() && result.compare.ok() &&
                   std::fabs(result.distribution.tvd - 0.04245) < 1e-6,
               detail);

        // the same totals concentrated in one campaign widen the interval,
        // because the resampling sees a campaign that behaved unlike the rest
        std::vector<CampaignRecord> lumpy;
        for (int i = 0; i < 9; ++i)
            lumpy.push_back(campaign({{1, 2000}}));
        lumpy.push_back(campaign({{1, 1001}, {2, 999}}));
        auto lumpyExec = makeExec(source, lumpy);
        addMatchingProbe(lumpyExec);
        auto lumpyResult = mlir_mracle::distributionOracleCompare(lumpyExec, oracleOptions());
        char lumpyDetail[192];
        std::snprintf(lumpyDetail, sizeof(lumpyDetail),
                      "tvd=%.5f ci=[%.5f, %.5f] verdict=%s",
                      lumpyResult.distribution.tvd, lumpyResult.distribution.ciLow,
                      lumpyResult.distribution.ciHigh,
                      lumpyResult.compare.ok()
                          ? (lumpyResult.compare.warn() ? "WARN" : "OK")
                          : "FAIL");
        report("150 -> 999 concentrated in one campaign",
               lumpyResult.distribution.ciHigh > result.distribution.ciHigh,
               lumpyDetail);
    }

    // where the two components hand over: the source's total count sits either
    // side of its own band (13.8 at 20,000 runs), and that alone decides which
    // component judges the same 999-occurrence shift
    {
        auto dense = [&] {
            std::vector<CampaignRecord> transformed;
            for (int i = 0; i < 9; ++i)
                transformed.push_back(campaign({{1, 1900}, {2, 100}}));
            transformed.push_back(campaign({{1, 1901}, {2, 99}}));
            return transformed;
        };
        auto sparse = [&](int64_t count) {
            std::vector<CampaignRecord> source;
            for (int i = 0; i < 10; ++i)
                source.push_back(i == 0 ? campaign({{1, 2000 - count}, {2, count}})
                                        : campaign({{1, 2000}}));
            return source;
        };

        auto quiet = makeExec(sparse(13), dense());
        addMatchingProbe(quiet);
        auto quietResult = mlir_mracle::distributionOracleCompare(quiet, oracleOptions());
        char quietDetail[192];
        std::snprintf(quietDetail, sizeof(quietDetail),
                      "tvd=%.5f violations=%zu verdict=%s",
                      quietResult.distribution.tvd, quietResult.support.size(),
                      quietResult.compare.ok()
                          ? (quietResult.compare.warn() ? "WARN" : "OK")
                          : "FAIL");
        report("13 -> 999 is judged categorically", !quietResult.compare.ok(),
               quietDetail);

        // the same 14 spread across campaigns, so only the handover is under test
        std::vector<CampaignRecord> even;
        for (int i = 0; i < 10; ++i)
            even.push_back(campaign({{1, 2000 - (i < 4 ? 2 : 1)}, {2, i < 4 ? 2 : 1}}));
        auto loud = makeExec(even, dense());
        addMatchingProbe(loud);
        auto loudResult = mlir_mracle::distributionOracleCompare(loud, oracleOptions());
        char loudDetail[192];
        std::snprintf(loudDetail, sizeof(loudDetail),
                      "tvd=%.5f violations=%zu verdict=%s",
                      loudResult.distribution.tvd, loudResult.support.size(),
                      loudResult.compare.ok()
                          ? (loudResult.compare.warn() ? "WARN" : "OK")
                          : "FAIL");
        report("14 -> 999 falls to the tolerance",
               loudResult.compare.ok() && !loudResult.compare.warn(),
               loudDetail);
    }

    // a rate movement inside a shared support is the distributional component's
    // judgement, not the support check's
    {
        auto exec = makeExec(repeat(campaign({{1, 1700}, {2, 300}}), 10),
                             repeat(campaign({{1, 1900}, {2, 100}}), 10));
        addMatchingProbe(exec);
        auto result = mlir_mracle::distributionOracleCompare(exec, oracleOptions());
        const bool pass = result.support.empty() &&
                          result.distribution.verdict == DistributionVerdict::Fail;
        report("shared support is judged by magnitude only", pass,
               "tvd=" + std::to_string(result.distribution.tvd) +
                   " violations=" + std::to_string(result.support.size()) +
                   " verdict=" +
                   std::to_string(static_cast<int>(result.distribution.verdict)));
    }

    // the relation reaches the verdict, not only the metric: dropping an
    // outcome the source produced is tolerated under subset and fails under
    // superset, and the mirror holds for an added outcome
    {
        auto dropped = makeExec(repeat(campaign({{1, 500}, {2, 500}}), 10),
                                repeat(campaign({{1, 1000}}), 10));
        addMatchingProbe(dropped);

        auto subset = oracleOptions();
        subset.relation = OutcomeRelation::Subset;
        expectResult("subset tolerates a dropped outcome",
                     mlir_mracle::distributionOracleCompare(dropped, subset), true, false,
                     false);

        auto superset = oracleOptions();
        superset.relation = OutcomeRelation::Superset;
        expectResult("superset fails on a dropped outcome",
                     mlir_mracle::distributionOracleCompare(dropped, superset), false, false,
                     false);

        auto added = makeExec(repeat(campaign({{1, 1000}}), 10),
                              repeat(campaign({{1, 500}, {2, 500}}), 10));
        addMatchingProbe(added);

        auto subsetAdd = oracleOptions();
        subsetAdd.relation = OutcomeRelation::Subset;
        expectResult("subset fails on an added outcome",
                     mlir_mracle::distributionOracleCompare(added, subsetAdd), false, false,
                     false);

        auto supersetAdd = oracleOptions();
        supersetAdd.relation = OutcomeRelation::Superset;
        expectResult("superset tolerates an added outcome",
                     mlir_mracle::distributionOracleCompare(added, supersetAdd), true, false,
                     false);
    }

    // a comparison with too few campaigns is inconclusive, never an ok
    {
        auto one = repeat(campaign({{1, 1000}}), 1);
        auto exec = makeExec(one, one);
        addMatchingProbe(exec);
        auto result = mlir_mracle::distributionOracleCompare(exec, oracleOptions(1));
        expectResult("single campaign is inconclusive", result, true, true,
                     false);

        bool plain = false;
        for (const auto &issue : result.compare.issues)
            if (issue.reason.rfind("programs differ by ", 0) == 0 &&
                issue.reason.find("% of probability") != std::string::npos)
                plain = true;
        report("undecided verdict is a one-liner", plain,
               result.compare.issues.empty()
                   ? "no issues"
                   : result.compare.issues.front().reason);
    }

    // a rate shift is reported in the units the tolerance uses
    {
        auto exec = makeExec(repeat(campaign({{1, 500}, {2, 500}}), 10),
                             repeat(campaign({{1, 900}, {2, 100}}), 10));
        addMatchingProbe(exec);
        auto result = mlir_mracle::distributionOracleCompare(exec, oracleOptions());
        bool plain = false;
        for (const auto &issue : result.compare.issues)
            if (issue.severity == mlir_mracle::IssueSeverity::Fail &&
                issue.reason.rfind("programs differ by ", 0) == 0 &&
                issue.reason.find("% of probability") != std::string::npos)
                plain = true;
        report("fail verdict is a one-liner", plain,
               result.compare.issues.empty()
                   ? "no issues"
                   : result.compare.issues.front().reason);
    }
}

} // namespace

int main() {
    testDiscrepancy();
    testRelationMasking();
    testSupportCheck();
    testUnseenMassBand();
    testStoppingRule();
    testBinomialTail();
    testOracleVerdicts();

    std::printf("%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILURES",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
