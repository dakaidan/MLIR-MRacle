#pragma once

#include "llvm/IR/Module.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mlir_mr {
struct RunInfo;
}

int executeLLVMModuleWithJIT(std::unique_ptr<llvm::Module> llvmModule,
                             mlir_mr::RunInfo *runInfo = nullptr);

// Compiles the module once and returns a callable that runs its "main".
// The callable resets all global state to the module's initial values before
// every call, so repeated calls behave like fresh JIT compilations.
// The returned vector holds one captured value per "main" result, in
// declaration order; values are zero-extended/sign-extended to int64_t.
// Returns nullptr on failure and fills *error.
std::function<std::vector<int64_t>()> compileLLVMModuleToFunction(
    std::unique_ptr<llvm::Module> module,
    std::string *error = nullptr);
