#pragma once

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/IR/LLVMContext.h"

namespace mlir_mr {

struct MLIRSetup {
    mlir::MLIRContext mlirContext;
    llvm::LLVMContext llvmContext;
    mlir::PassManager pm;

    MLIRSetup();
};

} // namespace mlir_mr
