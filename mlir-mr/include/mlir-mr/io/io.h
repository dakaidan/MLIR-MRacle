#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "llvm/IR/LLVMContext.h"
#include <string>

namespace mlir_mr {

std::string translateAndWriteToFile(mlir::ModuleOp module,
                                    llvm::LLVMContext &llvmContext,
                                    const std::string &filename);

} // namespace mlir_mr
