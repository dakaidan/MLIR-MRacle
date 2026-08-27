#pragma once

#include "llvm/IR/Module.h"
#include "llvm/Target/TargetOptions.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Compiles the module once and returns a callable that runs its "main"
// Options for TSan, varying JIT opt level
std::function<std::vector<int64_t>()> compileLLVMModuleToFunction(
    std::unique_ptr<llvm::Module> module,
    std::string *error = nullptr,
    bool enableTsan = false,
    int jitOptLevel = -1,
    llvm::BasicBlockSection bbSections = llvm::BasicBlockSection::None);
