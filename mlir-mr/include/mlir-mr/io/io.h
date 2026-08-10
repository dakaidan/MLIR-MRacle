#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "llvm/IR/Module.h"

#include <map>
#include <string>
#include <vector>

namespace mlir_mr {

// outcome frequencies per output position: index i maps a value seen at
// output i to how many runs produced it
using OutcomeCounts = std::vector<std::map<int64_t, int>>;

// IR printing utilities for --print-mlir and artifact logging
std::string dumpMLIR(mlir::ModuleOp module);
std::string dumpLLVM(llvm::Module &module);

// renders an outcome set as {v1, v2, ...}
std::string formatOutcomeSet(const std::map<int64_t, int> &counts);

// one affected output variable, gathered during comparison and rendered
// afterwards in a mode-dependent format
struct VariableIssue {
    std::string label;
    std::string originalSet;
    std::string transformedSet;
    std::vector<std::string> notes;  // novel/disappeared detail lines
    bool disjoint = false;
    bool hardFail = false;
};

struct CompareResult {
    bool ok = true;
    bool warn = false;
    std::string message;
};

// verbosity-aware rendering of the issues gathered during comparison
CompareResult renderComparison(int numRuns, bool verbose,
                               const std::vector<VariableIssue> &issues);

} // namespace mlir_mr
