#pragma once

#include "llvm/IR/Module.h"

#include <memory>

namespace mlir_mr {
struct RunInfo;
}

int executeLLVMModuleWithJIT(std::unique_ptr<llvm::Module> llvmModule,
                             mlir_mr::RunInfo *runInfo = nullptr);
