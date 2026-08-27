#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>
#include <set>
#include <utility>

namespace mlir_mracle {

// Relationship between the source and transformed outcome sets
// Used in oracle comparisons
enum class OutcomeRelation { Equality, Subset, Superset };

std::string outcomeRelationToString(OutcomeRelation relation);

bool outcomeRelationFromString(const std::string &s,
                               OutcomeRelation &relation);

// Severity of a verdict issue; no need for OK as it is implied by the absence of issues
enum class IssueSeverity { Fail, Warn };

std::string issueSeverityToString(IssueSeverity severity);

// Data object representing a single issue in a verdict for a comparison between two outcome sets
struct VerdictIssue {
    IssueSeverity severity = IssueSeverity::Warn;
    std::string outcome; // e.g. "[2]" or "var0=3"; empty when not applicable
    std::string reason;
};

// Result of a comparison between two outcome sets.
struct CompareResult {
    std::vector<VerdictIssue> issues; // the issues found in the comparison; empty when ok
    std::string note; // optional note for the verdict; empty when not applicable

    CompareResult() = default;

    // legacy convenience: builds a single synthetic issue when the verdict is
    // not OK, so the issues invariant always holds
    CompareResult(bool ok, bool warn, std::string message) {
        if (!ok)
            issues.push_back({IssueSeverity::Fail, "", std::move(message)});
        else if (warn)
            issues.push_back({IssueSeverity::Warn, "", std::move(message)});
    }

    // !ok implies fail status
    bool ok() const {
        return std::none_of(issues.begin(), issues.end(),
                            [](const VerdictIssue &i) {
                                return i.severity == IssueSeverity::Fail;
                            });
    }
    // warn means no fail issue and at least one warn issue
    bool warn() const {
        return ok() && std::any_of(issues.begin(), issues.end(),
                                   [](const VerdictIssue &i) {
                                       return i.severity == IssueSeverity::Warn;
                                   });
    }

    // "reason: outcome" per issue joined with "; ", then the note; kept for
    // legacy/internal consumers
    std::string message() const {
        std::string out;
        for (size_t i = 0; i < issues.size(); ++i) {
            if (i > 0)
                out += "; ";
            out += issues[i].reason;
            if (!issues[i].outcome.empty())
                out += ": " + issues[i].outcome;
        }
        if (!note.empty()) {
            if (!out.empty())
                out += "; ";
            out += note;
        }
        return out;
    }
};

// one ordered tuple of outputs produced by a single execution
using JointOutcome = std::vector<int64_t>;

// sorted unique joint outcomes observed for one program over a batch
struct ObservedOutcomeSet {
    std::vector<JointOutcome> outcomes; // sorted unique joint outcomes observed for one program over a batch
    std::vector<int64_t> counts; // number of occurrences for each joint outcome in outcomes; same order as outcomes
    size_t arity = 0;
    bool arityConsistent = true;
    int64_t totalRuns = 0; // total executions
};

// outcome-set comparison between the source and transformed programs
struct OutcomeSetResult {
    ObservedOutcomeSet source;
    ObservedOutcomeSet transformed;
    CompareResult compare;
};

// Identity of a compiled binary within one side of an agitation sweep
struct BinaryIdentity {
    std::string side;     // "source" | "transformed"
    int compileIndex = 0; // index to match same-config binaries across sides
    int jitOptLevel = -1; // CodeGen opt level used for this binary
};

} // namespace mlir_mracle
