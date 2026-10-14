// Regression tests for the distributional oracle. The decision rules are
// public API, so this links the library the pipeline runs instead of including
// a translation unit.
//
// The bootstrap is exercised only through stored campaign records, which is
// exactly the guarantee under test: resampling must never compile or execute a
// program.
#include "mlir-mracle/oracle/distribution.h"
#include "mlir-mracle/oracle/oracle.h"

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
using mlir_mracle::OracleOptions;
using mlir_mracle::OracleResult;
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

DistributionOptions options(double threshold = 0.05, int violationsAt = 5) {
    DistributionOptions opts;
    opts.tvdThreshold = threshold;
    opts.bootstrapReplicates = 200;
    opts.confidenceLevel = 0.95;
    opts.thresholdPct = violationsAt;
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

void expectResult(const char *name, const OracleResult &result, bool ok,
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

OracleOptions oracleOptions(size_t maxCampaigns = 10, double threshold = 0.05,
                            int violationsAt = 5) {
    OracleOptions opts;
    opts.maxCampaigns = maxCampaigns;
    opts.distribution = options(threshold, violationsAt);
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
                           DistributionVerdict::Bad, 1.0);
    }

    // hand-computed: P = [A: 1.0], Q = [A: 0.9, B: 0.1] -> tvd = 0.1
    {
        auto source = repeat(campaign({{1, 1000}}), 10);
        auto transformed = repeat(campaign({{1, 900}, {2, 100}}), 10);

        expectDistribution("hand-computed shift vs delta=0.05",
                           mlir_mracle::compareDistributions(
                               source, transformed, OutcomeRelation::Equality,
                               options(0.05)),
                           DistributionVerdict::Bad, 0.1);
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
                          result.verdict == DistributionVerdict::Bad;
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
                           DistributionVerdict::Bad, 1.0);
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
                       DistributionVerdict::Bad, 0.5);
    expectDistribution("subset ignores dropped mass",
                       mlir_mracle::compareDistributions(
                           source, transformed, OutcomeRelation::Subset,
                           options()),
                       DistributionVerdict::Ok, 0.0);
    expectDistribution("superset judges dropped mass",
                       mlir_mracle::compareDistributions(
                           source, transformed, OutcomeRelation::Superset,
                           options()),
                       DistributionVerdict::Bad, 0.5);

    // the transformed side adds an outcome the source never produced
    expectDistribution("subset judges added mass",
                       mlir_mracle::compareDistributions(
                           transformed, source, OutcomeRelation::Subset,
                           options()),
                       DistributionVerdict::Bad, 0.5);
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
    // the fluke band: one occurrence in ten thousand runs is ignored
    {
        std::vector<CampaignRecord> source;
        for (int i = 0; i < 10; ++i)
            source.push_back(i == 0 ? campaign({{1, 999}, {2, 1}})
                                    : campaign({{1, 1000}}));
        auto found = mlir_mracle::checkSupport(
            source, repeat(campaign({{1, 1000}}), 10),
            OutcomeRelation::Equality, options());
        if (found.size() != 1)
            report("fluke occurrence is ignored", false,
                   "violations=" + std::to_string(found.size()));
        else
            expectViolation("fluke occurrence is ignored", found, 0,
                            SupportVerdict::Ok, false, 1);
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

    // a missing outcome whose absence is already decisive but whose source rate
    // is under the detection threshold is also a candidate
    {
        std::vector<CampaignRecord> source;
        for (int i = 0; i < 10; ++i)
            source.push_back(i < 2 ? campaign({{1, 990}, {2, 10}})
                                   : campaign({{1, 1000}}));
        auto found = mlir_mracle::checkSupport(
            source, repeat(campaign({{1, 1000}}), 10),
            OutcomeRelation::Equality, options());
        if (found.size() != 1)
            report("decisive absence below threshold", false,
                   "violations=" + std::to_string(found.size()));
        else
            expectViolation("decisive absence below threshold", found, 0,
                            SupportVerdict::Candidate, true, 20);
    }

    // significant but below the detection threshold stays a candidate
    {
        auto source = repeat(campaign({{1, 990}, {2, 10}}), 10);
        auto found = mlir_mracle::checkSupport(
            source, repeat(campaign({{1, 1000}}), 10),
            OutcomeRelation::Equality, options());
        if (found.size() != 1)
            report("significant below threshold", false,
                   "violations=" + std::to_string(found.size()));
        else
            expectViolation("significant below threshold", found, 0,
                            SupportVerdict::Candidate, true, 100);
    }

    // clearing both bars: a fail naming the outcome
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

        auto small = mlir_mracle::checkSupport(source, withCount(13),
                                              OutcomeRelation::Equality,
                                              options());
        auto large = mlir_mracle::checkSupport(source, withCount(36),
                                              OutcomeRelation::Equality,
                                              options());
        const bool smallOk = small.size() == 1 && !small[0].significant;
        const bool largeOk = large.size() == 1 && large[0].significant;
        report("evidence bar sits at 36 occurrences", smallOk && largeOk,
               "13 significant=" + std::to_string(small.empty() ? -1 : small[0].significant) +
                   " 36 significant=" +
                   std::to_string(large.empty() ? -1 : large[0].significant));
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
                            SupportVerdict::Candidate, true, 100);
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
                              novel->verdict == SupportVerdict::Candidate;
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
                     mlir_mracle::oracleCompare(exec, opts), true, false, false);
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
        expectResult("live candidate at the cap is recorded, not failed",
                     mlir_mracle::oracleCompare(exec, opts), true, false, false);
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
        auto result = mlir_mracle::oracleCompare(exec, oracleOptions());
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
                     mlir_mracle::oracleCompare(exec, oracleOptions()), false,
                     false, false);
    }

    // a rate shift between outcomes both sides produce trips no support check,
    // so the discrepancy alone has to carry the verdict
    {
        auto exec = makeExec(repeat(campaign({{1, 500}, {2, 500}}), 10),
                             repeat(campaign({{1, 900}, {2, 100}}), 10));
        addMatchingProbe(exec);
        auto result = mlir_mracle::oracleCompare(exec, oracleOptions());
        expectResult("rate shift on shared outcomes fails", result, false, false,
                     false);
        report("shared outcomes raise no support violation",
               result.support.empty(),
               "violations=" + std::to_string(result.support.size()));
    }

    // a support violation fails while the discrepancy stays under tolerance:
    // the transformed side drops an outcome the source produced in 2% of runs
    // and the detection threshold is 1%
    {
        auto exec = makeExec(repeat(campaign({{1, 980}, {2, 20}}), 10),
                             repeat(campaign({{1, 1000}}), 10));
        addMatchingProbe(exec);
        auto result =
            mlir_mracle::oracleCompare(exec, oracleOptions(10, 0.05, 1));
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
                     mlir_mracle::oracleCompare(exec, oracleOptions(12)), true,
                     false, true);

        // a candidate too rare to confirm is not worth a verdict of its own: it
        // stays recorded as evidence and the run is ok
        auto capped = mlir_mracle::oracleCompare(exec, oracleOptions(10));
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
        auto result = mlir_mracle::oracleCompare(exec, oracleOptions());
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
                     mlir_mracle::oracleCompare(exec, oracleOptions()), false,
                     false, false);
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
                     mlir_mracle::oracleCompare(dropped, subset), true, false,
                     false);

        auto superset = oracleOptions();
        superset.relation = OutcomeRelation::Superset;
        expectResult("superset fails on a dropped outcome",
                     mlir_mracle::oracleCompare(dropped, superset), false, false,
                     false);

        auto added = makeExec(repeat(campaign({{1, 1000}}), 10),
                              repeat(campaign({{1, 500}, {2, 500}}), 10));
        addMatchingProbe(added);

        auto subsetAdd = oracleOptions();
        subsetAdd.relation = OutcomeRelation::Subset;
        expectResult("subset fails on an added outcome",
                     mlir_mracle::oracleCompare(added, subsetAdd), false, false,
                     false);

        auto supersetAdd = oracleOptions();
        supersetAdd.relation = OutcomeRelation::Superset;
        expectResult("superset tolerates an added outcome",
                     mlir_mracle::oracleCompare(added, supersetAdd), true, false,
                     false);
    }

    // a comparison with too few campaigns is inconclusive, never an ok
    {
        auto one = repeat(campaign({{1, 1000}}), 1);
        auto exec = makeExec(one, one);
        addMatchingProbe(exec);
        auto result = mlir_mracle::oracleCompare(exec, oracleOptions(1));
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
        auto result = mlir_mracle::oracleCompare(exec, oracleOptions());
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
