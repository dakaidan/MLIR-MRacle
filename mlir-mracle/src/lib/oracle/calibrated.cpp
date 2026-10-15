#include "mlir-mracle/oracle/calibrated.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <utility>

namespace mlir_mracle {
namespace {

int64_t totalCount(const OutcomeCounts &counts) {
    int64_t total = 0;
    for (const auto &entry : counts)
        total += entry.second;
    return total;
}

// Outcome counts of every usable campaign of one side, keeping the campaign
// boundary so the noise envelope can resample whole campaigns rather than
// individual executions.
std::vector<OutcomeCounts> campaignCounts(
    const std::vector<CompiledBinary> &binaries) {
    std::vector<OutcomeCounts> campaigns;
    campaigns.reserve(binaries.size());

    for (const auto &binary : binaries) {
        const ObservedOutcomeSet &set = binary.threadedTotal;
        OutcomeCounts counts;
        for (size_t i = 0; i < set.outcomes.size() && i < set.counts.size(); ++i)
            if (set.counts[i] > 0)
                counts[set.outcomes[i]] += set.counts[i];
        if (totalCount(counts) > 0)
            campaigns.push_back(std::move(counts));
    }
    return campaigns;
}

OutcomeCounts pool(const std::vector<OutcomeCounts> &campaigns) {
    OutcomeCounts pooled;
    for (const auto &campaign : campaigns)
        for (const auto &entry : campaign)
            pooled[entry.first] += entry.second;
    return pooled;
}

// Order statistic with linear interpolation between neighbours.
double quantile(const std::vector<double> &sorted, double fraction) {
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

// Scores of comparisons between two groups of campaigns drawn from the same
// pooled set: the envelope ordinary variability reaches at this budget.
std::vector<double> noiseEnvelope(const std::vector<OutcomeCounts> &campaigns,
                                  size_t sourceCampaigns, int permutations,
                                  uint32_t seed) {
    std::vector<size_t> order(campaigns.size());
    std::iota(order.begin(), order.end(), 0);

    std::mt19937 rng(seed);
    std::vector<double> scores;
    scores.reserve(static_cast<size_t>(permutations));

    std::vector<OutcomeCounts> left;
    std::vector<OutcomeCounts> right;
    for (int replicate = 0; replicate < permutations; ++replicate) {
        std::shuffle(order.begin(), order.end(), rng);
        left.clear();
        right.clear();
        for (size_t i = 0; i < order.size(); ++i) {
            if (i < sourceCampaigns)
                left.push_back(campaigns[order[i]]);
            else
                right.push_back(campaigns[order[i]]);
        }
        scores.push_back(squaredHellinger(pool(left), pool(right)));
    }

    std::sort(scores.begin(), scores.end());
    return scores;
}

std::string formatScore(double score) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.4f", score);
    return buffer;
}

VerdictIssue issue(IssueSeverity severity, std::string reason) {
    VerdictIssue out;
    out.severity = severity;
    out.reason = std::move(reason);
    return out;
}

} // namespace

double squaredHellinger(const OutcomeCounts &source,
                        const OutcomeCounts &transformed) {
    const double sourceTotal = static_cast<double>(totalCount(source));
    const double transformedTotal = static_cast<double>(totalCount(transformed));
    if (sourceTotal <= 0.0 || transformedTotal <= 0.0)
        return 1.0;

    // The overlap of the two profiles; every outcome outside it, on either side,
    // is counted as the distance rather than as a shared term.
    double overlap = 0.0;
    for (const auto &entry : source) {
        const auto other = transformed.find(entry.first);
        if (other == transformed.end())
            continue;
        overlap += std::sqrt((static_cast<double>(entry.second) / sourceTotal) *
                             (static_cast<double>(other->second) /
                              transformedTotal));
    }
    return std::clamp(1.0 - overlap, 0.0, 1.0);
}

CalibrationResult calibratedCompare(const ExecutionResult &exec,
                                    const CalibratedOptions &opts) {
    CalibrationResult out;

    const std::vector<OutcomeCounts> source = campaignCounts(exec.sourceBinaries);
    const std::vector<OutcomeCounts> transformed =
        campaignCounts(exec.transformedBinaries);
    const OutcomeCounts sourcePooled = pool(source);
    const OutcomeCounts transformedPooled = pool(transformed);

    out.sourceCampaigns = source.size();
    out.transformedCampaigns = transformed.size();
    out.sourceRuns = totalCount(sourcePooled);
    out.transformedRuns = totalCount(transformedPooled);
    out.sourceSupport = sourcePooled.size();
    out.transformedSupport = transformedPooled.size();
    for (const auto &entry : sourcePooled)
        if (!transformedPooled.count(entry.first))
            ++out.sourceOnly;
    for (const auto &entry : transformedPooled)
        if (!sourcePooled.count(entry.first))
            ++out.transformedOnly;
    out.minChange = opts.minChange;

    if (out.sourceRuns <= 0 || out.transformedRuns <= 0) {
        out.verdict = CalibrationVerdict::Fail;
        out.compare.issues.push_back(
            issue(IssueSeverity::Fail,
                  "no usable campaigns on one side of the comparison"));
        return out;
    }

    out.score = squaredHellinger(sourcePooled, transformedPooled);

    // The noise envelope comes from the campaigns of both sides, so it carries
    // the compilation and scheduling variability the comparison actually saw.
    std::vector<OutcomeCounts> everything = source;
    everything.insert(everything.end(), transformed.begin(), transformed.end());
    const int permutations = std::max(1, opts.permutations);
    const std::vector<double> null = noiseEnvelope(
        everything, source.size(), permutations, opts.seed);

    const int looks = std::max(1, opts.plannedLooks);
    const double alpha = std::clamp(opts.alphaTotal / looks, 1e-9, 1.0);
    out.permutations = permutations;
    out.noiseThreshold = quantile(null, 1.0 - alpha);
    out.decisionThreshold = std::max(out.noiseThreshold, opts.minChange);

    size_t atLeastAsExtreme = 0;
    for (double sample : null)
        if (sample >= out.score)
            ++atLeastAsExtreme;
    out.pValue = static_cast<double>(atLeastAsExtreme + 1) /
                 static_cast<double>(permutations + 1);

    if (out.score > out.decisionThreshold) {
        out.verdict = CalibrationVerdict::Fail;
        out.compare.issues.push_back(issue(
            IssueSeverity::Fail,
            "the outcome distributions differ beyond the calibrated bar: change "
            "score " +
                formatScore(out.score) + " against " +
                formatScore(out.decisionThreshold)));
        return out;
    }

    // Above the practical floor but inside the noise envelope: the budget may
    // still separate the two, so the comparison asks for more campaigns.
    if (out.score >= opts.minChange) {
        out.verdict = CalibrationVerdict::Inconclusive;
        out.needsRerun = true;
        out.compare.issues.push_back(issue(
            IssueSeverity::Warn,
            "the variants are not separated from their own variability at this "
            "budget: change score " +
                formatScore(out.score) + " against " +
                formatScore(out.decisionThreshold)));
        return out;
    }

    out.verdict = CalibrationVerdict::Ok;
    out.compare.note = "pooled outcome distributions differ by " +
                       formatScore(out.score) + ", below the bar of " +
                       formatScore(out.decisionThreshold);
    return out;
}

} // namespace mlir_mracle
