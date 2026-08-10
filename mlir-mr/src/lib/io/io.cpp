#include "mlir-mr/io/io.h"

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

} // namespace mlir_mr
