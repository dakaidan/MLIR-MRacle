#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir_mracle {

// Lower the given module to LLVM IR. The module must be in the LLVM dialect at this point.
mlir::LogicalResult lowerToLLVM(mlir::ModuleOp module, mlir::MLIRContext *ctx);

} // namespace mlir_mracle
