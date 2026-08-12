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

// Runs the given LLVM module in a JIT and returns the result of its "main" function.
int executeLLVMModuleWithJIT(std::unique_ptr<llvm::Module> llvmModule,
                             mlir_mr::RunInfo *runInfo = nullptr);

// Compiles the module once and returns a callable that runs its "main".
// When enableTsan is set, the module is instrumented with ThreadSanitizer,
// which perturbs memory-access scheduling and surfaces rare outcomes under
// concurrent execution. TSan requires the host binary to be built with
// -fsanitize=thread so the runtime is linked at startup.
std::function<std::vector<int64_t>()> compileLLVMModuleToFunction(
    std::unique_ptr<llvm::Module> module,
    std::string *error = nullptr,
    bool enableTsan = false);
