#pragma once

#include "mlir-mr/context/context.h"
#include "mlir-mr/core/core.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/JSON.h"

#include <map>
#include <string>
#include <vector>

namespace mlir_mr {

// IR printing utilities for --print-mlir and artifact logging
std::string dumpMLIR(mlir::ModuleOp module);
std::string dumpLLVM(llvm::Module &module);

// renders an outcome set as {v1, v2, ...}
std::string formatOutcomeSet(const std::map<int64_t, int> &counts);

// verbosity-aware rendering of the issues gathered during comparison
CompareResult renderComparison(int numRuns, bool verbose,
                               const std::vector<VariableIssue> &issues);

// renders a run's metadata as human-readable text
std::string formatRunInfo(const RunInfo &info);

// renders a run's metadata as a JSON object
llvm::json::Object runInfoToJson(const RunInfo &info, bool includeMLIR = false);

} // namespace mlir_mr
