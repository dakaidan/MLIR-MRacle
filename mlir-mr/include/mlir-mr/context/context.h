#pragma once

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/IR/LLVMContext.h"
#include <string>
#include <vector>

namespace mlir_mr {

// A single metamorphic transformation applied during a run.
// Used for logging and reporting in RunInfo.
struct AppliedTransformation {
    std::string name;
    std::string targetFunction;
};

// Main data object about a single run of the metamorphic testing pipeline.
struct RunInfo {
    int runNumber = 0;
    int seed = 42;
    std::string file;
    std::vector<std::string> requestedTransforms;
    std::vector<AppliedTransformation> appliedTransforms;
    bool transformApplied = false;
    std::string error;
    std::string warn;
    std::string mlirOutput;
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
