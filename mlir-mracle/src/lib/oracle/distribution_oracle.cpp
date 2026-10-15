#include "mlir-mracle/oracle/distribution_oracle.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <utility>

namespace mlir_mracle {
namespace {

// caps the number of issues one run reports, so a program with hundreds of
// deviating outcomes does not drown the artifacts in messages
constexpr size_t kMaxReportedViolations = 10;

// renders a joint outcome as [a, b, ...] for verdict messages
std::string formatOutcome(const JointOutcome &outcome) {
    std::string out = "[";
    for (size_t i = 0; i < outcome.size(); ++i) {
        if (i > 0)
            out += ", ";
        out += std::to_string(outcome[i]);
    }
    out += "]";
    return out;
}

// campaign records of one side, in campaign order
std::vector<CampaignRecord> campaignRecords(
    const std::vector<CompiledBinary> &binaries) {
    std::vector<CampaignRecord> records;
    records.reserve(binaries.size());

    for (const auto &binary : binaries) {
        CampaignRecord record;
        record.outcomes = binary.threadedTotal.outcomes;
        record.counts = binary.threadedTotal.counts;
        record.validRuns = binary.threadedTotal.totalRuns;
        record.usable = record.validRuns > 0;
        records.push_back(std::move(record));
    }
    return records;
}

// percentage with enough precision to stay meaningful for small differences
std::string formatPercent(double fraction) {
    char buffer[32];
    const double percent = fraction * 100.0;
    if (percent >= 1.0)
        std::snprintf(buffer, sizeof(buffer), "%.1f%%", percent);
    else
        std::snprintf(buffer, sizeof(buffer), "%.2f%%", percent);
    return buffer;
}

// the one-line statistic behind a distributional verdict
std::string describeProbability(const DistributionResult &result) {
    return "programs differ by " + formatPercent(result.tvd) +
           " of probability";
}

// the outcome behind a support verdict, named on the side that lacks it
std::string describeSupport(const SupportViolation &violation) {
    const std::string outcome = formatOutcome(violation.outcome);
    if (violation.transformedOnly)
        return "new outcome " + outcome + " not found in source";
    return "missing outcome " + outcome + " not found in transformed";
}

// the single-thread probe names both results when there is one per side
std::string describeProbe(const ObservedOutcomeSet &source,
                          const ObservedOutcomeSet &transformed) {
    if (source.outcomes.size() == 1 && transformed.outcomes.size() == 1)
        return "single-thread results differ (source " +
               formatOutcome(source.outcomes.front()) + ", transformed " +
               formatOutcome(transformed.outcomes.front()) + ")";
    return "single-thread results are not one outcome each (source " +
           std::to_string(source.outcomes.size()) + ", transformed " +
           std::to_string(transformed.outcomes.size()) + ")";
}

VerdictIssue failIssue(std::string outcome, std::string reason) {
    VerdictIssue issue;
    issue.severity = IssueSeverity::Fail;
    issue.outcome = std::move(outcome);
    issue.reason = std::move(reason);
    return issue;
}

VerdictIssue warnIssue(std::string outcome, std::string reason) {
    VerdictIssue issue;
    issue.severity = IssueSeverity::Warn;
    issue.outcome = std::move(outcome);
    issue.reason = std::move(reason);
    return issue;
}

} // namespace

DistributionOracleResult distributionOracleCompare(const ExecutionResult &exec,
                           const DistributionOracleOptions &opts) {
    DistributionOracleResult out;

    // at one thread both sides must produce exactly the same single outcome;
    // this runs before the distributional comparison and is judged on its own
    const ObservedOutcomeSet &sourceProbe = exec.sourceSingleThreadTotal;
    const ObservedOutcomeSet &transformedProbe = exec.transformedSingleThreadTotal;
    const bool probed = !sourceProbe.outcomes.empty() ||
                        !transformedProbe.outcomes.empty();
    if (probed) {
        const bool singleMatch = sourceProbe.outcomes.size() == 1 &&
                                 transformedProbe.outcomes.size() == 1 &&
                                 sourceProbe.outcomes.front() ==
                                     transformedProbe.outcomes.front();
        if (!singleMatch) {
            out.compare.issues.push_back(
                failIssue("", describeProbe(sourceProbe, transformedProbe)));
            return out;
        }
    }

    const std::vector<CampaignRecord> source = campaignRecords(exec.sourceBinaries);
    const std::vector<CampaignRecord> transformed =
        campaignRecords(exec.transformedBinaries);

    out.distribution = compareDistributions(source, transformed, opts.relation,
                                            opts.distribution);
    out.support = checkSupport(source, transformed, opts.relation,
                               opts.distribution);
    out.compare.note = describeProbability(out.distribution);

    // a violation that clears both the evidence bar and the detection
    // threshold is a hard failure, whatever the distributional verdict says
    std::vector<VerdictIssue> failIssues;
    size_t failures = 0;
    for (const auto &violation : out.support) {
        if (violation.verdict != SupportVerdict::Fail)
            continue;
        ++failures;
        if (failures <= kMaxReportedViolations)
            failIssues.push_back(
                failIssue(formatOutcome(violation.outcome),
                          describeSupport(violation)));
    }

    // the discrepancy itself exceeding the tolerance is a hard failure as well,
    // and it is the only signal for a rate shift between outcomes both sides
    // produce
    if (out.distribution.verdict == DistributionVerdict::Fail)
        failIssues.push_back(
            failIssue("", describeProbability(out.distribution)));

    if (!failIssues.empty()) {
        if (failures > kMaxReportedViolations)
            failIssues.push_back(failIssue(
                "", std::to_string(failures - kMaxReportedViolations) +
                        " more outcomes differ the same way"));
        out.compare.issues = std::move(failIssues);
        return out;
    }

    const size_t collected = std::max(exec.sourceBinaries.size(),
                                      exec.transformedBinaries.size());
    const bool inconclusive =
        out.distribution.verdict == DistributionVerdict::Inconclusive;
    // A candidate is only worth sampling for if the budget still reachable could
    // establish it; one that never could is recorded and left alone.
    const size_t pendingEstablishable = static_cast<size_t>(std::count_if(
        out.support.begin(), out.support.end(),
        [](const SupportViolation &violation) {
            return (violation.verdict == SupportVerdict::Candidate ||
                    violation.verdict == SupportVerdict::Warn) &&
                   violation.establishable;
        }));

    // keep sampling while either component is unresolved and the budget allows
    if ((inconclusive || pendingEstablishable > 0) &&
        collected < opts.maxCampaigns) {
        out.needsRerun = true;
        out.compare.note =
            describeProbability(out.distribution) + "; collecting more campaigns";
        return out;
    }

    // The budget is spent. Anything the other variant's own sampling makes
    // unlikely is reported, however rare it is; only the genuinely ambiguous
    // middle, where the evidence points nowhere yet, stays recorded and silent.
    size_t warned = 0;
    for (const auto &violation : out.support) {
        if (violation.verdict != SupportVerdict::Warn)
            continue;
        ++warned;
        if (warned <= kMaxReportedViolations)
            out.compare.issues.push_back(
                warnIssue(formatOutcome(violation.outcome),
                          describeSupport(violation)));
    }
    if (warned > kMaxReportedViolations)
        out.compare.issues.push_back(
            warnIssue("", std::to_string(warned - kMaxReportedViolations) +
                              " more unlikely outcomes"));

    if (inconclusive)
        out.compare.issues.push_back(
            warnIssue("", describeProbability(out.distribution)));

    return out;
}

} // namespace mlir_mracle
