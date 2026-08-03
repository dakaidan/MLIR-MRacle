#include "mlir-mr/backend/jit/jit.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/Support/TargetSelect.h"
#include <llvm/IR/Constants.h>
#include "llvm/ExecutionEngine/Orc/Core.h"

#include <utility>

int executeLLVMModuleWithJIT(std::unique_ptr<llvm::Module> llvmModule) {

    // Init the native target and its components
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();


    // Set up the JIT compiler or error out if it fails
    auto jitOrErr = llvm::orc::LLJITBuilder().create();
    if (!jitOrErr) {
        llvm::errs() << "Failed to create JIT: "
                    << llvm::toString(jitOrErr.takeError()) << "\n";
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

    auto addr = sym->getValue();
    auto *fn = llvm::jitTargetAddressToPointer<int32_t (*)()>(addr);
    int32_t ret = fn();
    return ret;
}

