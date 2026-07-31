#include "mlir-mr/backend/jit.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/Support/TargetSelect.h"
#include <utility>

int executeLLVMModuleWithJIT(std::unique_ptr<llvm::Module> llvmModule) {
    // Set up the JIT compiler or error out if it fails
    auto jitOrErr = llvm::orc::LLJITBuilder().create();
    if (!jitOrErr) {
        llvm::errs() << "Failed to create JIT\n";
        return 1;
    }

    // Move the JIT instance out now that we know it was created successfully
    auto jit = std::move(*jitOrErr);

    // Add the LLVM module to the JIT and check for errors
    if (auto err = jit->addIRModule(
            llvm::orc::ThreadSafeModule(std::move(llvmModule),
                                         std::make_unique<llvm::LLVMContext>()))) {
        llvm::errs() << "Failed to add module\n";
        return 1;
    }
    
    // Look up the "main" function
    auto sym = jit->lookup("main");
    if (!sym) {
        llvm::errs() << "main not found\n";
        return 1;
    }

    // TODO: exact byte outputs
    // For now, assume the main function has the signature int main() and call it
    auto *fn = sym->toPtr<int32_t(*)()>();
    int32_t ret = fn();
    return ret;
}
