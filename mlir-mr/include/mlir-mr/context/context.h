#pragma once

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/JSON.h"
#include <string>

namespace mlir_mr {

struct RunInfo {
    int runNumber = 0;
    int seed = 42;
    std::string file;
    std::string requestedTransforms;
    std::string appliedTransform;
    std::string targetFunction;
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

    MLIRSetup(int seed = 42, int runNumber = 0, std::string transform = "");
};

} // namespace mlir_mr
