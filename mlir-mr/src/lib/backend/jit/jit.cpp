#include "mlir-mr/backend/jit/jit.h"
#include "mlir-mr/context/context.h"

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/TargetSelect.h"

#include <cstdint>
#include <mutex>
#include <utility>
#include <string>
#include <vector>

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

// number of scalar results produced by the module's "main" function
static unsigned getResultCount(llvm::Module &module) {
    auto *mainFn = module.getFunction("main");
    if (!mainFn || mainFn->isDeclaration())
        return 0;
    if (auto *st = llvm::dyn_cast<llvm::StructType>(mainFn->getReturnType()))
        if (!st->isOpaque() && st->getNumElements() > 0)
            return st->getNumElements();
    return 1;
}

// Adds a "__mlir_mr_run" wrapper that calls "main" and stores every result
// into the "__mlir_mr_results" global (a buffer of int64_t, one slot per
// result, in declaration order). This avoids ABI-specific struct-return
// handling on the C++ side.
static void addResultCaptureFunction(llvm::Module &module) {
    auto &ctx = module.getContext();
    auto *mainFn = module.getFunction("main");
    if (!mainFn || mainFn->isDeclaration())
        return;

    unsigned numResults = getResultCount(module);
    if (numResults == 0)
        return;

    auto *i64Ty = llvm::Type::getInt64Ty(ctx);
    auto *bufferTy = llvm::ArrayType::get(i64Ty, numResults);
    // External linkage: the buffer is only written inside the module, so
    // internal linkage would let the optimizer drop it as a dead global.
    auto *buffer = new llvm::GlobalVariable(
        module, bufferTy, false, llvm::GlobalValue::ExternalLinkage,
        llvm::ConstantAggregateZero::get(bufferTy), "__mlir_mr_results");

    auto *runFn = llvm::Function::Create(
        llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), false),
        llvm::GlobalValue::ExternalLinkage, "__mlir_mr_run", &module);

    llvm::IRBuilder<> builder(ctx);
    llvm::BasicBlock *entry = llvm::BasicBlock::Create(ctx, "entry", runFn);
    builder.SetInsertPoint(entry);

    llvm::Value *ret = builder.CreateCall(mainFn);
    for (unsigned i = 0; i < numResults; ++i) {
        llvm::Value *v = numResults == 1
                             ? ret
                             : builder.CreateExtractValue(ret, i);
        if (v->getType()->isIntegerTy()) {
            if (v->getType()->getIntegerBitWidth() != 64)
                v = builder.CreateSExtOrTrunc(v, i64Ty);
        } else if (v->getType()->isPointerTy()) {
            v = builder.CreatePtrToInt(v, i64Ty);
        } else {
            continue; // non-integer scalar results are not captured
        }
        llvm::Value *slot = builder.CreateInBoundsGEP(
            bufferTy, buffer,
            {builder.getInt32(0), builder.getInt32(i)});
        builder.CreateStore(v, slot);
    }
    builder.CreateRetVoid();
}

// For one-time compilation, and keeps the compiled code alive
std::function<std::vector<int64_t>()> compileLLVMModuleToFunction(
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

    unsigned numResults = module ? getResultCount(*module) : 0;
    if (module) {
        addStateResetFunction(*module);
        addResultCaptureFunction(*module);
    }

    if (auto err = jit->addIRModule(
            llvm::orc::ThreadSafeModule(std::move(module),
                                        std::make_unique<llvm::LLVMContext>()))) {
        if (error)
            *error = "failed to add module: " + llvm::toString(std::move(err));
        return nullptr;
    }

    auto runSym = jit->lookup("__mlir_mr_run");
    if (!runSym) {
        if (error)
            *error = "main not found";
        return nullptr;
    }
    auto *runFn =
        llvm::jitTargetAddressToPointer<void (*)()>(runSym->getValue());

    void (*resetFn)() = nullptr;
    if (auto resetSym = jit->lookup("__mlir_mr_reset_state"))
        resetFn =
            llvm::jitTargetAddressToPointer<void (*)()>(resetSym->getValue());

    auto resultsSym = jit->lookup("__mlir_mr_results");
    if (!resultsSym) {
        if (error)
            *error = "result buffer not found";
        return nullptr;
    }
    auto *resultsBuf =
        llvm::jitTargetAddressToPointer<int64_t *>(resultsSym->getValue());

    return [jit, runFn, resetFn, resultsBuf, numResults]() -> std::vector<int64_t> {
        if (resetFn)
            resetFn();
        runFn();
        return std::vector<int64_t>(resultsBuf, resultsBuf + numResults);
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
    auto results = fn();
    return results.empty() ? 0 : static_cast<int>(results[0]);
}

