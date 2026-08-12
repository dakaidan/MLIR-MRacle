#pragma once

#include "mlir-mr/context/context.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/JSON.h"

#include <string>

namespace mlir_mr {

// IR printing utilities for --print-mlir and artifact logging
std::string dumpMLIR(mlir::ModuleOp module);
std::string dumpLLVM(llvm::Module &module);

// renders a run's metadata as human-readable text
std::string formatRunInfo(const RunInfo &info);

// renders a single Fisher result as a JSON object
llvm::json::Object fisherResultToJson(const FisherResult &fr);

// parses a Fisher result from a JSON object; returns false on malformed input
bool fisherResultFromJson(const llvm::json::Object &o, FisherResult &fr);

// renders a run's metadata as a JSON object
llvm::json::Object runInfoToJson(const RunInfo &info, bool includeMLIR = false);

} // namespace mlir_mr
