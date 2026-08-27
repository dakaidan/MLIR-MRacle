#pragma once

#include "mlir-mracle/core/run_info.h"
#include "mlir-mracle/core/types.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/IR/LLVMContext.h"
#include <string>

namespace mlir_mracle {

// registers all dialects and LLVM IR translations shared by every pipeline
void initializeMLIRContext(mlir::MLIRContext &ctx);

struct MLIRSetup {
    mlir::MLIRContext mlirContext;
    llvm::LLVMContext llvmContext;
    mlir::PassManager pm;
    RunInfo runInfo; // shared run info that is accumulated during a campaign

    MLIRSetup(int seed = 42, int runNumber = 0, std::string transform = "",
              int maxApply = 1, std::string model = "");
};

} // namespace mlir_mracle
