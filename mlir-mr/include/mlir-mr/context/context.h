#pragma once

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/IR/LLVMContext.h"
#include <string>

namespace mlir_mr {

struct MLIRSetup {
    mlir::MLIRContext mlirContext;
    llvm::LLVMContext llvmContext;
    mlir::PassManager pm;

    MLIRSetup(int seed = 42, const std::string &transforms = "", bool debug = false);
};

} // namespace mlir_mr
