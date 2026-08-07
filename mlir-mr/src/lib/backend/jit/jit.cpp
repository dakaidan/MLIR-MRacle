#include "mlir-mr/backend/jit/jit.h"
#include "mlir-mr/context/context.h"

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/TargetSelect.h"

#include <mutex>
#include <utility>
#include <string>

static void initNativeTargetOnce() {
    static std::once_flag flag;
    std::call_once(flag, []() {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();
    });
}

// Resetting global state is done to ensure that each run of the JIT starts with the same initial state
static void addStateResetFunction(llvm::Module &module) {
    auto &ctx = module.getContext();
    auto *fn = llvm::Function::Create(
        llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), false),
        llvm::GlobalValue::ExternalLinkage, "__mlir_mr_reset_state", &module);

    llvm::IRBuilder<> builder(ctx);
    builder.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", fn));

    for (auto &global : module.globals()) {
        if (global.isDeclaration() || !global.hasInitializer() ||
            global.getName().starts_with("llvm."))
            continue;
        builder.CreateStore(global.getInitializer(), &global);
    }

    builder.CreateRetVoid();
}

// For one-time compilation, and keeps the compiled code alive
std::function<int32_t()> compileLLVMModuleToFunction(
    std::unique_ptr<llvm::Module> module,
    std::string *error) {

    if (error)
        error->clear();
    initNativeTargetOnce();

    auto jitOrErr = llvm::orc::LLJITBuilder().create();
    if (!jitOrErr) {
        if (error)
            *error = "failed to create JIT: " +
                     llvm::toString(jitOrErr.takeError());
        return nullptr;
    }

    // Adopt the JIT instance; the shared_ptr keeps compiled code alive as long
    // as the returned callable is referenced.
    std::shared_ptr<llvm::orc::LLJIT> jit(jitOrErr->release());

    if (module)
        addStateResetFunction(*module);

    if (auto err = jit->addIRModule(
            llvm::orc::ThreadSafeModule(std::move(module),
                                        std::make_unique<llvm::LLVMContext>()))) {
        if (error)
            *error = "failed to add module: " + llvm::toString(std::move(err));
        return nullptr;
    }

    auto sym = jit->lookup("main");
    if (!sym) {
        if (error)
            *error = "main not found";
        return nullptr;
    }

    auto *fn = llvm::jitTargetAddressToPointer<int32_t (*)()>(sym->getValue());

    void (*resetFn)() = nullptr;
    if (auto resetSym = jit->lookup("__mlir_mr_reset_state"))
        resetFn =
            llvm::jitTargetAddressToPointer<void (*)()>(resetSym->getValue());

    return [jit, fn, resetFn]() -> int32_t {
        if (resetFn)
            resetFn();
        return fn();
    };
}

// actual execution of the JIT'd code, with error handling
int executeLLVMModuleWithJIT(std::unique_ptr<llvm::Module> llvmModule,
                             mlir_mr::RunInfo *runInfo) {
    std::string error;
    auto fn = compileLLVMModuleToFunction(std::move(llvmModule), &error);
    if (!fn) {
        if (runInfo)
            runInfo->error = error;
        return 1;
    }
    if (runInfo)
        runInfo->error.clear();
    return fn();
}

