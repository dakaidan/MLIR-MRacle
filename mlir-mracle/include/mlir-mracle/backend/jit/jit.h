#pragma once

#include "llvm/IR/Module.h"
#include "llvm/Target/TargetOptions.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Compiles the module once and returns a callable that runs its "main".
// When enableTsan is set, the module is instrumented with ThreadSanitizer,
// which perturbs memory-access scheduling and surfaces rare outcomes under
// concurrent execution. TSan requires the host binary to be built with
// -fsanitize=thread so the runtime is linked at startup. jitOptLevel selects
// the LLVM CodeGen opt level (0-3); -1 keeps the JIT's default. bbSections
// overrides the code layout; BasicBlockSection::All emits every basic block
// into its own section, preserving the module's block order in the generated
// machine code (the default is None). It is honoured only on ELF targets;
// on MachO/COFF the request is ignored because LLVM implements per-basic-
// block sections for ELF alone.
std::function<std::vector<int64_t>()> compileLLVMModuleToFunction(
    std::unique_ptr<llvm::Module> module,
    std::string *error = nullptr,
    bool enableTsan = false,
    int jitOptLevel = -1,
    llvm::BasicBlockSection bbSections = llvm::BasicBlockSection::None);
