#include "mlir-mr/backend/jit/jit.h"
#include "mlir-mr/context/context.h"

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/Instrumentation/ThreadSanitizer.h"

#include <cstdint>
#include <mutex>
#include <utility>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

// initialise the native target and its asm printer/parser exactly once, even if multiple
static void initNativeTarget() {
    static std::once_flag flag;
    std::call_once(flag, []() {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();
    });
}

// The TSan runtime must be linked into the host binary at startup
// (-fsanitize=thread): compiler-rt's TSan cannot be dlopen'd late on macOS,
// and its Linux runtime is not on the loader path. JIT'd code then resolves
// __tsan_* symbols against the already-initialised runtime.
static bool loadTsanRuntime(std::string *error) {
    if (llvm::sys::DynamicLibrary::SearchForAddressOfSymbol("__tsan_init"))
        return true;
    if (error)
        *error = "TSan runtime not linked; rebuild with -fsanitize=thread";
    return false;
}

// Instruments the module with ThreadSanitizer. TSan instruments every memory
// access, perturbing scheduling so rare outcomes surface under concurrent
// execution.
static bool instrumentWithTsan(llvm::Module &module, std::string *error) {
    if (!loadTsanRuntime(error))
        return false;

    llvm::LoopAnalysisManager LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager CGAM;
    llvm::ModuleAnalysisManager MAM;

    llvm::PassBuilder PB;
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    llvm::ModulePassManager MPM;
    MPM.addPass(
        llvm::createModuleToFunctionPassAdaptor(llvm::ThreadSanitizerPass()));
    MPM.addPass(llvm::ModuleThreadSanitizerPass());
    MPM.run(module, MAM);
    return true;
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
        // constants are immutable; storing to one is rejected by the verifier
        // and would be UB at runtime
        if (global.isDeclaration() || global.isConstant() ||
            !global.hasInitializer() || global.getName().starts_with("llvm."))
            continue;
        builder.CreateStore(global.getInitializer(), &global);
    }

    builder.CreateRetVoid();
}

// returns the number of scalar results produced by the module's "main" function
static unsigned getResultCount(llvm::Module &module) {
    auto *mainFn = module.getFunction("main");
    if (!mainFn || mainFn->isDeclaration())
        return 0;
    if (auto *st = llvm::dyn_cast<llvm::StructType>(mainFn->getReturnType()))
        if (!st->isOpaque() && st->getNumElements() > 0)
            return st->getNumElements();
    return 1;
}

// Adds a "__mlir_mr_run(i64* results)" wrapper that calls "main" and stores
// every result into the caller-provided buffer. Each concurrent caller passes
// its own buffer, so result collection is race-free even with TSan active.
static void addResultCaptureFunction(llvm::Module &module) {
    auto &ctx = module.getContext();
    auto *mainFn = module.getFunction("main");
    if (!mainFn || mainFn->isDeclaration())
        return;

    unsigned numResults = getResultCount(module);
    if (numResults == 0)
        return;

    auto *i64Ty = llvm::Type::getInt64Ty(ctx);
    auto *i64PtrTy = llvm::PointerType::get(ctx, 0);

    auto *runFn = llvm::Function::Create(
        llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), {i64PtrTy},
                                false),
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
        llvm::Value *slot = builder.CreateGEP(
            i64Ty, runFn->getArg(0), builder.getInt32(i));
        builder.CreateStore(v, slot);
    }
    builder.CreateRetVoid();
}

// One-time compilation to LLVM
std::function<std::vector<int64_t>()> compileLLVMModuleToFunction(
    std::unique_ptr<llvm::Module> module,
    std::string *error,
    bool enableTsan,
    int jitOptLevel,
    llvm::BasicBlockSection bbSections) {

    if (error)
        error->clear();
    initNativeTarget();

    llvm::orc::LLJITBuilder builder;
    if (jitOptLevel >= 0 || bbSections != llvm::BasicBlockSection::None) {
        auto jtmb = llvm::orc::JITTargetMachineBuilder::detectHost();
        if (jtmb) {
            if (jitOptLevel >= 0)
                jtmb->setCodeGenOptLevel(
                    static_cast<llvm::CodeGenOptLevel>(jitOptLevel));
            if (bbSections != llvm::BasicBlockSection::None) {
                auto opts = jtmb->getOptions();
                opts.BBSections = bbSections;
                jtmb->setOptions(opts);
            }
            builder.setJITTargetMachineBuilder(std::move(*jtmb));
        }
    }
    auto jitOrErr = builder.create();
    if (!jitOrErr) {
        if (error)
            *error = "failed to create JIT: " +
                     llvm::toString(jitOrErr.takeError());
        return nullptr;
    }

    // shared_ptr keeps compiled code alive as long as the returned callable is referenced.
    std::shared_ptr<llvm::orc::LLJIT> jit(jitOrErr->release());

    if (!module) {
        if (error)
            *error = "main not found";
        return nullptr;
    }

    unsigned numResults = getResultCount(*module);
    addStateResetFunction(*module);
    addResultCaptureFunction(*module);
    if (enableTsan && !instrumentWithTsan(*module, error))
        return nullptr;

    auto tsm = llvm::orc::cloneExternalModuleToContext(
        *module,
        llvm::orc::ThreadSafeContext(std::make_unique<llvm::LLVMContext>()));
    if (auto err = jit->addIRModule(std::move(tsm))) {
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
        llvm::jitTargetAddressToPointer<void (*)(int64_t *)>(runSym->getValue());

    void (*resetFn)() = nullptr;
    if (auto resetSym = jit->lookup("__mlir_mr_reset_state"))
        resetFn =
            llvm::jitTargetAddressToPointer<void (*)()>(resetSym->getValue());

    // each invocation gets its own stack buffer, so concurrent callers never
    // share mutable state through the results channel
    return [jit, runFn, resetFn, numResults]() -> std::vector<int64_t> {
        std::vector<int64_t> results(numResults);
        if (resetFn)
            resetFn();
        runFn(results.data());
        return results;
    };
}

// actual execution of the JIT'd code, with error handling
int executeLLVMModuleWithJIT(std::unique_ptr<llvm::Module> llvmModule,
                             mlir_mr::RunInfo *runInfo,
                             int jitOptLevel) {
    std::string error;
    auto fn = compileLLVMModuleToFunction(std::move(llvmModule), &error,
                                          false, jitOptLevel);
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

