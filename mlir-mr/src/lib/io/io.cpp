#include "mlir-mr/io/io.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir_mr {

std::string dumpMLIR(mlir::ModuleOp module) {
    std::string buf;
    llvm::raw_string_ostream os(buf);
    module.print(os);
    os.flush();
    return buf;
}

std::string dumpLLVM(llvm::Module &module) {
    std::string buf;
    llvm::raw_string_ostream os(buf);
    module.print(os, nullptr);
    os.flush();
    return buf;
}

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

CompareResult renderComparison(int numRuns, bool verbose,
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
                entry += "\n  disjoint over " + std::to_string(numRuns) +
                         " runs";
            for (auto &note : issue.notes)
                entry += "\n  " + note;
        } else if (issue.disjoint) {
            entry = issue.label +
                    ": outcome sets are completely disjoint over " +
                    std::to_string(numRuns) + " runs";
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

    if (!failBody.empty())
        return {false, true,
                (verbose ? "=== SUMMARY: behavioural change detected over " +
                               std::to_string(numRuns) + " runs ===\n"
                         : "") +
                    failBody +
                    (warnBody.empty() ? "" : (verbose ? "\n" : " [") +
                                                   warnBody +
                                                   (verbose ? "" : "]"))};
    return {true, true,
            verbose
                ? "=== SUMMARY: outcomes disappeared over " +
                      std::to_string(numRuns) + " runs ===\n" + warnBody
                : "outcomes disappeared over " + std::to_string(numRuns) +
                      " runs [" + warnBody + "]"};
}

std::string formatRunInfo(const RunInfo &info) {
    std::string buf;
    llvm::raw_string_ostream os(buf);
    os << "run: " << info.runNumber << "\n";
    os << "seed: " << info.seed << "\n";
    os << "file: " << info.file << "\n";
    os << "requested-transforms: ";
    if (info.requestedTransforms.empty())
        os << "all\n";
    else
        os << llvm::join(info.requestedTransforms, ",") << "\n";
    if (info.transformApplied)
        for (const auto &at : info.appliedTransforms)
            os << "applied-transformation: " << at.name
               << " in function '" << at.targetFunction << "'\n";
    else
        os << "applied-transformation: none\n";
    if (!info.error.empty())
        os << "error: " << info.error << "\n";
    if (!info.warn.empty())
        os << "warn: " << info.warn << "\n";
    return buf;
}

llvm::json::Object runInfoToJson(const RunInfo &info, bool includeMLIR) {
    llvm::json::Object obj;
    obj["run"] = info.runNumber;
    obj["seed"] = info.seed;
    obj["file"] = info.file;
    llvm::json::Array requested;
    if (info.requestedTransforms.empty())
        requested.push_back("all");
    else
        for (const auto &r : info.requestedTransforms)
            requested.push_back(r);
    obj["requested_transforms"] = std::move(requested);
    llvm::json::Array applied;
    for (const auto &at : info.appliedTransforms) {
        llvm::json::Object entry;
        entry["name"] = at.name;
        entry["target_function"] = at.targetFunction;
        applied.push_back(std::move(entry));
    }
    obj["applied_transforms"] = std::move(applied);
    obj["transform_applied"] = info.transformApplied;
    if (!info.error.empty())
        obj["error"] = info.error;
    if (!info.warn.empty())
        obj["warn"] = info.warn;
    if (includeMLIR)
        obj["mlir_output"] = info.mlirOutput;
    return obj;
}

} // namespace mlir_mr
