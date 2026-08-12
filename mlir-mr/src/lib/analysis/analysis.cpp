#include "mlir-mr/analysis/analysis.h"

#include <string>
#include <vector>

namespace mlir_mr {

std::string formatOutcomeSet(const std::map<int64_t, int> &counts) {
    std::string s = "{";
    bool first = true;
    for (auto &[val, _] : counts) {
        if (!first)
            s += ", ";
        first = false;
        s += std::to_string(val);
    }
    s += "}";
    return s;
}

CompareResult renderComparison(int originalRuns, int transformedRuns,
                               bool verbose,
                               const std::vector<VariableIssue> &issues) {
    std::string failBody, warnBody;
    const std::string sep = verbose ? "\n" : "; ";
    for (auto &issue : issues) {
        std::string entry;
        if (verbose) {
            // one line per outcome set, then the issue lines
            entry = issue.label + ":\n  original outcome set: " +
                    issue.originalSet + "\n  transformed outcome set: " +
                    issue.transformedSet;
            if (issue.disjoint)
                entry += "\n  disjoint over original " +
                         std::to_string(originalRuns) + " / transformed " +
                         std::to_string(transformedRuns) + " runs";
            for (auto &note : issue.notes)
                entry += "\n  " + note;
        } else if (issue.disjoint) {
            entry = issue.label +
                    ": outcome sets are completely disjoint over original " +
                    std::to_string(originalRuns) + " / transformed " +
                    std::to_string(transformedRuns) + " runs";
        } else {
            for (size_t n = 0; n < issue.notes.size(); ++n)
                entry += (n == 0 ? "" : "; ") + issue.label + ": " +
                         issue.notes[n];
        }
        std::string &dst = issue.hardFail ? failBody : warnBody;
        if (!dst.empty())
            dst += sep;
        dst += entry;
    }

    // the summary header is always emitted; only the per-outcome detail is
    // gated on verbose so the warn/fail reason is visible by default
    if (!failBody.empty())
        return {false, true,
                "=== SUMMARY: behavioural change detected over original " +
                    std::to_string(originalRuns) + " / transformed " +
                    std::to_string(transformedRuns) + " runs ===\n" +
                    failBody +
                    (warnBody.empty() ? "" : "\n" + warnBody)};
    return {true, true,
            "=== SUMMARY: outcomes disappeared over original " +
                std::to_string(originalRuns) + " / transformed " +
                std::to_string(transformedRuns) + " runs ===\n" + warnBody};
}

CompareResult compareOutcomes(const OutcomeCounts &originalCounts,
                              const OutcomeCounts &transformedCounts,
                              int originalRuns, int transformedRuns,
                              bool verbose, Relation relation) {
    // check that the two modules produce the same number of outputs
    if (originalCounts.size() != transformedCounts.size())
        return {false, false,
                "result arity mismatch: original produces " +
                    std::to_string(originalCounts.size()) +
                    " outputs, transformed produces " +
                    std::to_string(transformedCounts.size())};

    std::vector<VariableIssue> issues;
    bool identical = true;

    const bool forbidNovel =
        relation == Relation::Equality || relation == Relation::Subset;
    const bool forbidMissing =
        relation == Relation::Equality || relation == Relation::Superset;

    for (size_t i = 0; i < originalCounts.size(); ++i) {
        const auto &orig = originalCounts[i];
        const auto &transf = transformedCounts[i];

        if (orig != transf)
            identical = false;

        // label the output variable for reporting purposes
        std::string label = originalCounts.size() == 1
                                ? "output"
                                : "output " + std::to_string(i);

        bool anyOverlap = false;
        bool anyNovel = false;

        // check for any overlap or novel outcomes between the original and transformed outcome sets
        for (auto &[val, _] : transf)
            if (orig.count(val) > 0)
                anyOverlap = true;
            else
                anyNovel = true;

        VariableIssue issue;
        issue.label = label;
        issue.originalSet = formatOutcomeSet(orig);
        issue.transformedSet = formatOutcomeSet(transf);

        // sets disjoint -> hard fail
        if (forbidNovel && anyNovel && !anyOverlap) {
            issue.disjoint = true;
            issue.hardFail = true;
            issues.push_back(std::move(issue));
            continue;
        }

        // novel outcome appears over the threshold -> hard fail, else warn;
        // the count is measured against the transformed program's own total
        if (forbidNovel) {
            for (auto &[val, c] : transf) {
                if (orig.count(val) > 0)
                    continue;
                std::string note =
                    "novel outcome " + std::to_string(val) + " (" +
                    std::to_string(c) + "/" +
                    std::to_string(transformedRuns) + " runs) " +
                    (c >= transformedRuns * kNovelOutcomeFrequency
                         ? "exceeds " +
                               std::to_string(static_cast<int>(
                                   kNovelOutcomeFrequency * 100)) +
                               "% of runs"
                         : "below failure threshold");
                if (c >= transformedRuns * kNovelOutcomeFrequency)
                    issue.hardFail = true;
                issue.notes.push_back(note);
            }
        }

        // absent outcome disappears over the threshold -> hard fail, else
        // warn; the count refers to the source program's own total
        if (forbidMissing) {
            for (auto &[val, c] : orig) {
                if (transf.count(val) > 0)
                    continue;
                issue.notes.push_back(
                    "outcome " + std::to_string(val) +
                    " present in original (" + std::to_string(c) + "/" +
                    std::to_string(originalRuns) +
                    " runs) but absent in transformed");
                if (c >= originalRuns * kNovelOutcomeFrequency)
                    issue.hardFail = true;
            }
        }

        if (!issue.notes.empty())
            issues.push_back(std::move(issue));
    }

    // if there are no issues, return a summary message indicating whether the outcome sets are identical or changed
    if (issues.empty()) {
        std::string msg =
            identical
                ? "identical outcome maps (original " +
                      std::to_string(originalRuns) + " runs, transformed " +
                      std::to_string(transformedRuns) + " runs)"
                : "outcome sets changed (original " +
                      std::to_string(originalRuns) + " runs, transformed " +
                      std::to_string(transformedRuns) + " runs)";
        return {true, false, msg};
    }

    return renderComparison(originalRuns, transformedRuns, verbose, issues);
}

} // namespace mlir_mr
