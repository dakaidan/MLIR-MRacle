// Validation battery for the noise-calibrated comparison: every case the plan
// asks for, measured on campaigns built to the same shape the pipeline collects.
#include "mlir-mracle/oracle/calibrated.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using mlir_mracle::CalibratedOptions;
using mlir_mracle::CalibrationResult;
using mlir_mracle::CalibrationVerdict;
using mlir_mracle::CompiledBinary;
using mlir_mracle::ExecutionResult;
using mlir_mracle::JointOutcome;
using mlir_mracle::OutcomeCounts;

// nominal frequency of each outcome value on one side
using Mix = std::vector<std::pair<int64_t, double>>;

int failures = 0;

void report(const char *name, bool pass, const std::string &detail) {
    std::printf("[%s] %-52s %s\n", pass ? "PASS" : "FAIL", name, detail.c_str());
    if (!pass)
        ++failures;
}

std::string verdictName(CalibrationVerdict verdict) {
    switch (verdict) {
    case CalibrationVerdict::Ok:
        return "OK";
    case CalibrationVerdict::Fail:
        return "FAIL";
    case CalibrationVerdict::Inconclusive:
        return "INCONCLUSIVE";
    }
    return "?";
}

// Campaigns of one side. `spread` is the campaign-to-campaign standard
// deviation of the outcome frequencies, standing in for build and scheduling
// variability; zero keeps every campaign identical in expectation.
std::vector<CompiledBinary> buildCampaigns(const Mix &mix, int64_t perCampaign,
                                           size_t count, double spread,
                                           std::mt19937 &rng) {
    std::vector<CompiledBinary> binaries;
    for (size_t c = 0; c < count; ++c) {
        std::vector<std::pair<int64_t, double>> rates;
        double assigned = 0.0;
        for (size_t i = 0; i + 1 < mix.size(); ++i) {
            double rate = mix[i].second;
            if (spread > 0.0) {
                std::normal_distribution<double> jitter(mix[i].second, spread);
                rate = std::clamp(jitter(rng), 0.0, 1.0 - assigned);
            }
            rates.push_back({mix[i].first, rate});
            assigned += rate;
        }
        if (!mix.empty())
            rates.push_back({mix.back().first, std::max(0.0, 1.0 - assigned)});

        CompiledBinary binary;
        for (const auto &[value, rate] : rates) {
            std::binomial_distribution<int64_t> draw(perCampaign, rate);
            const int64_t n = draw(rng);
            if (n <= 0)
                continue;
            binary.threadedTotal.outcomes.push_back({value});
            binary.threadedTotal.counts.push_back(n);
            binary.threadedTotal.totalRuns += n;
        }
        binary.threadedTotal.arity = 1;
        binary.threadedTotal.arityConsistent = true;
        binaries.push_back(std::move(binary));
    }
    return binaries;
}

CalibratedOptions options(size_t campaigns = 10, int64_t perCampaign = 1000,
                          double minChange = 0.005, int permutations = 400,
                          uint32_t seed = 7) {
    CalibratedOptions opts;
    opts.minChange = minChange;
    opts.alphaTotal = 0.05;
    opts.plannedLooks = 1;
    opts.permutations = permutations;
    opts.seed = seed;
    opts.maxCampaigns = campaigns;
    (void)perCampaign;
    return opts;
}

CalibrationResult judge(const Mix &source, const Mix &transformed,
                        const CalibratedOptions &opts, size_t campaigns = 10,
                        int64_t perCampaign = 1000, double sourceSpread = 0.0,
                        double transformedSpread = 0.0, uint32_t seed = 7) {
    std::mt19937 rng(seed);
    ExecutionResult exec;
    exec.sourceBinaries = buildCampaigns(source, perCampaign, campaigns,
                                         sourceSpread, rng);
    exec.transformedBinaries = buildCampaigns(transformed, perCampaign, campaigns,
                                              transformedSpread, rng);
    return mlir_mracle::calibratedCompare(exec, opts);
}

Mix shifted(Mix base, int64_t to, double mass) {
    Mix out;
    for (const auto &[value, p] : base)
        out.push_back({value, value == to ? p + mass : p - mass});
    return out;
}

std::string describe(const CalibrationResult &result) {
    char buffer[192];
    std::snprintf(buffer, sizeof(buffer),
                  "%s  score %.4f bar %.4f (noise %.4f) p %.3f",
                  verdictName(result.verdict).c_str(), result.score,
                  result.decisionThreshold, result.noiseThreshold,
                  result.pValue);
    return buffer;
}

// --- the distance itself -----------------------------------------------------

void testDistance() {
    const OutcomeCounts a = {{{0}, 70}, {{1}, 30}};
    const OutcomeCounts b = {{{0}, 65}, {{1}, 35}};
    const OutcomeCounts c = {{{0}, 95}, {{1}, 5}};
    const OutcomeCounts d = {{{0}, 100}};

    report("identical distributions score zero",
           std::fabs(mlir_mracle::squaredHellinger(a, a)) < 1e-12, "0.000000");
    report("disjoint supports score one",
           std::fabs(mlir_mracle::squaredHellinger(d, {{{7}, 100}}) - 1.0) < 1e-12,
           "1.000000");
    report("common outcome 70/30 -> 65/35",
           std::fabs(mlir_mracle::squaredHellinger(a, b) - 0.001426) < 1e-6,
           "0.001426");
    report("rare outcome 0.75% -> 5% outweighs it",
           std::fabs(mlir_mracle::squaredHellinger({{{0}, 9925}, {{1}, 75}}, c) -
                     0.009618) < 1e-6 &&
               mlir_mracle::squaredHellinger({{{0}, 9925}, {{1}, 75}}, c) >
                   6.0 * mlir_mracle::squaredHellinger(a, b),
           "0.009618 vs 0.001426");
}

// --- the plan's behaviour list ----------------------------------------------

void testRareEmergence() {
    const Mix source = {{0, 0.9925}, {1, 0.0075}};
    const CalibrationResult result =
        judge(source, shifted(source, 1, 0.0424), options(10, 2000), 10, 2000);
    report("rare outcome 0.75% -> 5% fails", result.verdict == CalibrationVerdict::Fail,
           describe(result));

    // an outcome the source never produced at all
    const Mix empty = {{0, 1.0}};
    const Mix appeared = {{0, 0.985}, {7, 0.015}};
    const CalibrationResult fromNothing =
        judge(empty, appeared, options(10, 2000), 10, 2000);
    report("outcome absent from the source fails when it emerges",
           fromNothing.verdict == CalibrationVerdict::Fail, describe(fromNothing));
}

void testCommonOutcomeChanges() {
    const Mix source = {{0, 0.70}, {1, 0.30}};
    const CalibrationResult small =
        judge(source, shifted(source, 1, 0.025), options(10, 2000), 10, 2000);
    report("common outcome 30% -> 32.5% passes",
           small.verdict == CalibrationVerdict::Ok, describe(small));

    const CalibrationResult moderate =
        judge(source, shifted(source, 1, 0.05), options(10, 2000), 10, 2000);
    report("common outcome 30% -> 35% passes",
           moderate.verdict == CalibrationVerdict::Ok, describe(moderate));

    const CalibrationResult large =
        judge(source, shifted(source, 1, 0.15), options(10, 2000), 10, 2000);
    report("common outcome 30% -> 45% fails",
           large.verdict == CalibrationVerdict::Fail, describe(large));
}

void testUnseenOutcomes() {
    const Mix source = {{0, 0.70}, {1, 0.30}};
    const CalibrationResult tiny = judge(
        source, {{0, 0.6965}, {1, 0.2985}, {7, 0.005}}, options(10, 2000), 10, 2000);
    report("extra outcome at 0.5% stays below the floor",
           tiny.verdict == CalibrationVerdict::Ok, describe(tiny));

    const CalibrationResult visible = judge(
        source, {{0, 0.686}, {1, 0.294}, {7, 0.02}}, options(10, 2000), 10, 2000);
    report("extra outcome at 2% fails",
           visible.verdict == CalibrationVerdict::Fail, describe(visible));

    // one percentage point spread over fifty distinct singletons
    Mix singletons = {{0, 0.99}};
    for (int64_t i = 0; i < 50; ++i)
        singletons.push_back({100 + i, 0.0002});
    const CalibrationResult scattered =
        judge({{0, 1.0}}, singletons, options(10, 2000), 10, 2000);
    report("fifty singletons carrying 1% land on the floor",
           std::fabs(scattered.score - 0.005) < 0.0015 &&
               scattered.verdict != CalibrationVerdict::Ok,
           describe(scattered));

    Mix faint = {{0, 0.9975}};
    for (int64_t i = 0; i < 50; ++i)
        faint.push_back({100 + i, 0.00005});
    const CalibrationResult faintResult =
        judge({{0, 1.0}}, faint, options(10, 2000), 10, 2000);
    report("fifty singletons carrying 0.25% pass",
           faintResult.verdict == CalibrationVerdict::Ok, describe(faintResult));
}

void testUnequalBudgets() {
    const Mix source = {{0, 0.70}, {1, 0.30}};
    const CalibrationResult result =
        judge(source, source, options(10, 2000), 10, 2000, 0.0, 0.0);
    report("identical sides pass", result.verdict == CalibrationVerdict::Ok,
           describe(result));

    // five campaigns against ten at the same rates: the smaller side is noisier
    CalibratedOptions opts = options(5, 2000);
    opts.maxCampaigns = 5;
    std::mt19937 rng(3);
    ExecutionResult exec;
    exec.sourceBinaries = buildCampaigns(source, 2000, 10, 0.0, rng);
    exec.transformedBinaries = buildCampaigns(source, 2000, 5, 0.0, rng);
    const CalibrationResult uneven = mlir_mracle::calibratedCompare(exec, opts);
    report("5 campaigns against 10 at the same rates pass",
           uneven.verdict == CalibrationVerdict::Ok, describe(uneven));
}

void testCampaignVariability() {
    const Mix source = {{0, 0.70}, {1, 0.30}};

    const CalibrationResult quiet =
        judge(source, source, options(10, 1000), 10, 1000, 0.0, 0.0);
    const CalibrationResult noisy = judge(source, source, options(10, 1000), 10,
                                          1000, 0.02, 0.02);

    report("a run whose campaigns vary build to build still passes",
           noisy.verdict == CalibrationVerdict::Ok, describe(noisy));
    report("the envelope grows with campaign variability",
           noisy.noiseThreshold > 3.0 * quiet.noiseThreshold,
           "noise " + std::to_string(quiet.noiseThreshold) + " -> " +
               std::to_string(noisy.noiseThreshold));

    // the same pooled distributions, different variability: no separate
    // variability detector, so this is expected to pass
    const CalibrationResult asymmetric =
        judge(source, source, options(10, 1000), 10, 1000, 0.0, 0.03);
    report("differing variability with equal pooled rates passes",
           asymmetric.verdict == CalibrationVerdict::Ok, describe(asymmetric));
    report("the noisier side widens the envelope",
           noisy.noiseThreshold > quiet.noiseThreshold &&
               asymmetric.noiseThreshold > quiet.noiseThreshold,
           "asymmetric noise " + std::to_string(asymmetric.noiseThreshold));
}

void testUndecidedAsksForMore() {
    const Mix source = {{0, 0.70}, {1, 0.30}};

    // A floor below the noise the campaigns themselves carry: any score between
    // the two is undecided rather than equivalence.
    const CalibrationResult result =
        judge(source, source, options(10, 1000, 1e-6), 10, 1000, 0.02, 0.02);
    report("a score inside the noise envelope asks for more campaigns",
           result.needsRerun && result.verdict == CalibrationVerdict::Inconclusive,
           describe(result));
    report("undecided is never reported as equivalence",
           result.verdict != CalibrationVerdict::Ok, describe(result));

    // the same rule across draws: undecided exactly when the score sits between
    // the floor and the bar
    bool invariant = true;
    for (int i = 0; i < 25; ++i) {
        const CalibrationResult draw = judge(
            source, source,
            options(10, 1000, 1e-6, 200, static_cast<uint32_t>(500 + i)), 10,
            1000, 0.02, 0.02, static_cast<uint32_t>(900 + i));
        const bool expected = draw.score >= draw.minChange &&
                              draw.score <= draw.decisionThreshold;
        if (draw.needsRerun != expected)
            invariant = false;
    }
    report("undecided holds across draws", invariant, "25 draws");
}

void testFalseAlarmRate() {
    const Mix source = {{0, 0.70}, {1, 0.30}};
    const int comparisons = 100;
    int failed = 0;
    int reruns = 0;
    for (int i = 0; i < comparisons; ++i) {
        const CalibrationResult result = judge(
            source, source, options(10, 1000, 0.005, 200,
                                    static_cast<uint32_t>(1000 + i)),
            10, 1000, 0.02, 0.02, static_cast<uint32_t>(2000 + i));
        if (result.verdict == CalibrationVerdict::Fail)
            ++failed;
        if (result.needsRerun)
            ++reruns;
    }
    char detail[128];
    std::snprintf(detail, sizeof(detail), "%d/%d false alarms, %d undecided",
                  failed, comparisons, reruns);
    report("known-equivalent comparisons stay quiet", failed <= 5, detail);
}

void testDetectionRate() {
    const Mix source = {{0, 0.9925}, {1, 0.0075}};
    const Mix transformed = shifted(source, 1, 0.0424);
    int detected = 0;
    const int comparisons = 40;
    for (int i = 0; i < comparisons; ++i) {
        const CalibrationResult result = judge(
            source, transformed,
            options(10, 1000, 0.005, 200, static_cast<uint32_t>(3000 + i)), 10,
            1000, 0.01, 0.01, static_cast<uint32_t>(4000 + i));
        if (result.verdict == CalibrationVerdict::Fail)
            ++detected;
    }
    char detail[128];
    std::snprintf(detail, sizeof(detail), "%d/%d detected", detected, comparisons);
    report("a rare emergence is detected in a batch", detected >= 38, detail);
}

void testDeterminismAndGuards() {
    const Mix source = {{0, 0.70}, {1, 0.30}};
    const Mix transformed = shifted(source, 1, 0.05);
    const CalibrationResult first = judge(source, transformed, options());
    const CalibrationResult second = judge(source, transformed, options());
    report("the same campaigns give the same verdict",
           first.verdict == second.verdict && first.score == second.score &&
               first.noiseThreshold == second.noiseThreshold,
           describe(first));

    ExecutionResult empty;
    const CalibrationResult guarded =
        mlir_mracle::calibratedCompare(empty, options());
    report("a side without campaigns fails loudly",
           guarded.verdict == CalibrationVerdict::Fail && !guarded.compare.issues.empty(),
           guarded.compare.issues.empty() ? "" : guarded.compare.issues.front().reason);
}

} // namespace

int main() {
    std::printf("== distance ==\n");
    testDistance();
    std::printf("\n== rare-outcome behaviour ==\n");
    testRareEmergence();
    std::printf("\n== common-outcome behaviour ==\n");
    testCommonOutcomeChanges();
    std::printf("\n== unseen outcomes and singletons ==\n");
    testUnseenOutcomes();
    std::printf("\n== sample sizes and variability ==\n");
    testUnequalBudgets();
    testCampaignVariability();
    std::printf("\n== deciding ==\n");
    testUndecidedAsksForMore();
    std::printf("\n== batches ==\n");
    testFalseAlarmRate();
    testDetectionRate();
    std::printf("\n== determinism and guards ==\n");
    testDeterminismAndGuards();

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILURES",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
