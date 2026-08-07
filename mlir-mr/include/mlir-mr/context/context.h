#pragma once

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/JSON.h"
#include <string>
#include <vector>

namespace mlir_mr {

// A single metamorphic transformation applied during a run.
struct AppliedTransformation {
    std::string name;
    std::string targetFunction;
};

struct RunInfo {
    int runNumber = 0;
    int seed = 42;
    std::string file;
    std::vector<std::string> requestedTransforms;
    std::vector<AppliedTransformation> appliedTransforms;
    bool transformApplied = false;
    std::string error;
    std::string mlirOutput;

    std::string toString() const;
    llvm::json::Object toJson(bool includeMLIR = false) const;
};

struct MLIRSetup {
    mlir::MLIRContext mlirContext;
    llvm::LLVMContext llvmContext;
    mlir::PassManager pm;
    RunInfo runInfo;

    MLIRSetup(int seed = 42, int runNumber = 0, std::string transform = "",
              int maxApply = 1);
};

} // namespace mlir_mr
