#include "conquer/backend/cpu/cpu_target.h"
#include "conquer/core/logging.h"
#include "conquer/backend/capabilities.h"
#include "conquer/backend/cpu/cpu_packer.h"
#include "conquer/core/utils.h"

#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/IndexToLLVM/IndexToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/TosaToArith/TosaToArith.h>
#include <mlir/Conversion/TosaToLinalg/TosaToLinalg.h>
#include <mlir/Conversion/TosaToTensor/TosaToTensor.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Dialect/Tosa/Transforms/Passes.h>
#include <mlir/ExecutionEngine/CRunnerUtils.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>

#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Support/TargetSelect.h>

#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/Mangling.h>


#include "conquer/core/context.h"

#include <cmath>

#undef DEBUG_TYPE
#define DEBUG_TYPE "conquer-cpu-target"

extern "C" float conquer_rsqrtf_shim(float x) {
    return 1.0f / std::sqrt(x);
}

extern "C" double conquer_rsqrt_shim(double x) {
    return 1.0 / std::sqrt(x);
}

llvm::Error conquer::CPUTarget::compile(llvm::StringRef bytecode) {
    this->local_context_ = createMLIRContext();

    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(bytecode, local_context_.get());

    if (!module) {
        return llvm::make_error<llvm::StringError>("Failed to parse bytecode into MLIR module.", llvm::inconvertibleErrorCode());
    }

    return compile(module.release());
}

llvm::Error conquer::CPUTarget::compile(mlir::Operation *module) {
    L_INFO("Starting lowering pipeline for CPU target.");

    auto *context = module->getContext();
    context->disableMultithreading();
    context->printOpOnDiagnostic(true);

    mlir::PassManager pm(context);
    pm.enableVerifier(true);
    pm.enableCrashReproducerGeneration("conquer_crash.mlir");


    // -------------------------------------------------------------------------
    // Initial clean-up
    // -------------------------------------------------------------------------
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    // -------------------------------------------------------------------------
    // TOSA pre-processing
    // -------------------------------------------------------------------------
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaMakeBroadcastablePass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaOptionalDecompositionsPass());

    // -------------------------------------------------------------------------
    // TOSA -> Linalg
    // -------------------------------------------------------------------------
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalgNamed());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalg());

    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    // -------------------------------------------------------------------------
    // TOSA -> Arith / Tensor
    // -------------------------------------------------------------------------
    {
        mlir::TosaToArithPassOptions tosaToArithOpts;
        tosaToArithOpts.includeApplyRescale = true;
        pm.addNestedPass<mlir::func::FuncOp>(mlir::createTosaToArithPass(tosaToArithOpts));
    }
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createTosaToTensorPass());

    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    // -------------------------------------------------------------------------
    // Bufferization
    // IMPORTANT:
    // Use identity-layout memrefs at function boundaries. This avoids turning
    // entry-point tensors into fully dynamic strided memrefs, which is what
    // tends to trigger extract_strided_metadata + affine.apply in reshape-heavy
    // cases like your Transformer_Head_Block.
    // -------------------------------------------------------------------------
    {
        mlir::bufferization::OneShotBufferizePassOptions bufferizeOpts;
        bufferizeOpts.bufferizeFunctionBoundaries = true;
        bufferizeOpts.functionBoundaryTypeConversion =
            mlir::bufferization::LayoutMapOption::IdentityLayoutMap;
        bufferizeOpts.unknownTypeConversion =
            mlir::bufferization::LayoutMapOption::IdentityLayoutMap;

        pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufferizeOpts));
    }

    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    // -------------------------------------------------------------------------
    // Structured -> loops / low-level memref form
    // -------------------------------------------------------------------------
    pm.addPass(mlir::memref::createExpandOpsPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());

    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    // Expand view-like memref ops before final LLVM lowering.
    pm.addPass(mlir::memref::createExpandStridedMetadataPass());
    pm.addPass(mlir::createLowerAffinePass());

    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    // -------------------------------------------------------------------------
    // SCF -> CF
    // -------------------------------------------------------------------------
    pm.addPass(mlir::createSCFToControlFlowPass());

    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    // -------------------------------------------------------------------------
    // Mark only the true entrypoint with C interface.
    // Do NOT do this for every function by default; it can create unnecessary
    // wrappers for private/external helper funcs like your stats hook.
    // -------------------------------------------------------------------------
    module->walk([&](mlir::func::FuncOp func) {
        if (func.getName() == "main") {
            func->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(context));
        }
    });

    // -------------------------------------------------------------------------
    // Lower to LLVM dialect
    // Keep this tail close to what you already had; the main fix above is the
    // identity-layout boundary bufferization.
    // -------------------------------------------------------------------------
    pm.addPass(mlir::createConvertMathToLibmPass());

    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createConvertIndexToLLVMPass());
    pm.addPass(mlir::createConvertMathToLLVMPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertVectorToLLVMPass());
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());

    pm.addPass(mlir::createReconcileUnrealizedCastsPass());

    L_DEBUG("Running CPU target lowering pipeline.");

    if (mlir::failed(pm.run(module))) {
        llvm::errs() << "\n[conquer] Lowering failed. Module at failure point:\n";
        module->print(llvm::errs());
        llvm::errs() << "\n";
        return llvm::make_error<llvm::StringError>(
            "Failed to lower module to LLVM IR.",
            llvm::inconvertibleErrorCode());
    }

    this->loweredModule = module;
    return llvm::Error::success();
}

llvm::Error conquer::CPUTarget::save_compiled_module_to_file(const std::string &filename) const {
    if (!loweredModule) {
        L_DEBUG("No lowered module to save. Did you call compile()?");
        return llvm::make_error<llvm::StringError>("No lowered module to save. Did you call compile()?", llvm::inconvertibleErrorCode());
    }

    std::ofstream outFile(filename);
    if (!outFile) {
        L_DEBUG("Failed to open file for writing: " << filename);
        return llvm::make_error<llvm::StringError>("Failed to open file for writing: " + filename, llvm::inconvertibleErrorCode());
    }

    outFile << *loweredModule;
    outFile.close();

    L_INFO("Lowered module saved to: " << filename);
    return llvm::Error::success();
}

llvm::Error conquer::CPUTarget::load_compiled_module_from_file(const std::string &filename) {
    std::ifstream inFile(filename);
    if (!inFile) {
        L_DEBUG("Failed to open file for reading: " << filename);
        return llvm::make_error<llvm::StringError>("Failed to open file for reading: " + filename, llvm::inconvertibleErrorCode());
    }

    std::string moduleContent((std::istreambuf_iterator<char>(inFile)), std::istreambuf_iterator<char>());
    inFile.close();

    local_context_ = createMLIRContext();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(moduleContent, local_context_.get());
    if (!module) {
        L_DEBUG("Failed to parse MLIR module from file: " << filename);
        return llvm::make_error<llvm::StringError>("Failed to parse MLIR module from file: " + filename, llvm::inconvertibleErrorCode());
    }

    this->loweredModule = mlir::OwningOpRef<mlir::Operation *>(module.release());
    L_INFO("Lowered module loaded from: " << filename);
    return llvm::Error::success();
}

llvm::Error conquer::CPUTarget::execute(const std::vector<TensorView> &inputs, const std::vector<TensorView> &outputs) {
    L_INFO("Starting execution on CPU target.");

    llvm::Error init_err = llvm::Error::success();
    std::call_once(engine_init_flag_, [this, &init_err]() {
        init_err = build_execution_engine();
    });

    if (init_err) {
        return init_err;
    }

    if (!engine_) {
        return llvm::make_error<llvm::StringError>(
            "Execution engine initialization failed on a previous call.",
            llvm::inconvertibleErrorCode());
    }

    std::vector<std::vector<char>> descriptorBuffers;
    descriptorBuffers.reserve(inputs.size() + outputs.size());

    std::vector<void *> args;
    args.reserve(inputs.size() + outputs.size());

    std::vector<char> outputStructBuffer;

    if (!outputs.empty()) {
        if (outputs.size() == 1) {
            auto desc = buildMemRefDescriptor(nullptr, outputs[0].shape);
            descriptorBuffers.push_back(std::move(desc));
            args.push_back(descriptorBuffers.back().data());
        } else {
            for (const auto &output : outputs) {
                auto desc = buildMemRefDescriptor(nullptr, output.shape);
                outputStructBuffer.insert(outputStructBuffer.end(), desc.begin(), desc.end());
            }
            args.push_back(outputStructBuffer.data());
        }
    }

    for (const auto &input : inputs) {
        auto desc = buildMemRefDescriptor(input.data, input.shape);
        descriptorBuffers.push_back(std::move(desc));
        args.push_back(descriptorBuffers.back().data());
    }

    std::vector<void *> packedArgs;
    packedArgs.reserve(args.size());
    for (auto &argPtr : args) {
        packedArgs.push_back(&argPtr);
    }

    L_DEBUG("Invoking with " << packedArgs.size() << " arguments.");

    if (llvm::Error invocationResult = engine_->invokePacked("_mlir_ciface_main", packedArgs)) {
        return invocationResult;
    }

    if (!outputs.empty()) {
        const char *descBytes = static_cast<char *>(args[0]);
        size_t currentOffset = 0;

        for (size_t i = 0; i < outputs.size(); ++i) {
            void *resultPtr = nullptr;
            std::memcpy(&resultPtr, descBytes + currentOffset + sizeof(void *), sizeof(void *));

            if (resultPtr) {
                size_t numElements = 1;
                for (const auto s : outputs[i].shape)
                    numElements *= s;

                std::memcpy(outputs[i].data, resultPtr, numElements * sizeof(float));
            } else {
                return llvm::make_error<llvm::StringError>("JIT returned null pointer", llvm::inconvertibleErrorCode());
            }

            size_t rank = outputs[i].shape.size();
            size_t descSize = 24 + (16 * rank);
            currentOffset += descSize;
        }
    }

    return llvm::Error::success();
}

conquer::HardwareCapability conquer::CPUTarget::query_capability() const { return query_cpu_capability(); }

llvm::Error conquer::CPUTarget::build_execution_engine() {
    if (!loweredModule) {
        return llvm::make_error<llvm::StringError>(
            "No module to JIT compile.",
            llvm::inconvertibleErrorCode());
    }

    static bool initialised = ([] {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();
        return true;
    })();

    if (!initialised) {
        return llvm::make_error<llvm::StringError>(
            "Failed to init LLVM target.",
            llvm::inconvertibleErrorCode());
    }

    mlir::ExecutionEngineOptions engineOptions;
    auto opt = mlir::makeOptimizingTransformer(2, 0, nullptr);
    engineOptions.transformer = opt;


    // We can pass the lowered module directly. No need to clone it unless
    // you plan on mutating it further, which you aren't.
    auto maybeEngine = mlir::ExecutionEngine::create(loweredModule.get(), engineOptions);
    if (!maybeEngine) {
        return maybeEngine.takeError();
    }

    this->engine_ = std::move(maybeEngine.get());

    engine_->registerSymbols(
        [](llvm::orc::MangleAndInterner interner) -> llvm::orc::SymbolMap {
            llvm::orc::SymbolMap symbols;

            auto makeSymbol = [](auto fnPtr) -> llvm::orc::ExecutorSymbolDef {
                return llvm::orc::ExecutorSymbolDef(
                    llvm::orc::ExecutorAddr::fromPtr(fnPtr),
                    llvm::JITSymbolFlags::Exported |
                    llvm::JITSymbolFlags::Callable);
            };

            symbols[interner("memrefCopy")] = makeSymbol(+memrefCopy);
            symbols[interner("rsqrtf")] = makeSymbol(+conquer_rsqrtf_shim);
            symbols[interner("rsqrt")]  = makeSymbol(+conquer_rsqrt_shim);

            return symbols;
        });

    return llvm::Error::success();
}
