
#include "mlir-mracle/legacy/source_memo.h"

#include "mlir-mracle/backend/jit/jit.h"
#include "mlir-mracle/io/cache.h"
#include "mlir-mracle/pipeline/common/pipeline_common.h"

#include "llvm/Transforms/Utils/Cloning.h"

#include <filesystem>
#include <string>
#include <vector>
#include <map>

namespace mlir_mracle {

std::map<std::string, SourceMemo> gSourceMemo;

// invalidates a baseline entry (memory and disk) once the per-run retest cap
// is reached with outcomes still missing: the entry can no longer learn, so
// the next run of this file recollects from scratch and can take in new
// possible values
void clearBaseline(SourceMemo &memo, const std::string &baseKey, int reps) {
    memo.baselines.erase(baseKey);
    if (!cachingEnabled())
        return;
    std::error_code ec;
    std::filesystem::remove(
        baselineCachePath(memo.sourceHash, baseKey, reps).string(), ec);
}

// lowers and compiles the source module once, caching the plain binary (and
// holding the module for an on-demand TSan compile) so later runs of the same
// file skip recompilation; the lowered bitcode is also persisted so later
// processes of the same source skip lowering and translation entirely
bool memoizeSource(mlir::MLIRContext &mlirCtx, mlir::ModuleOp module,
                   const std::string &sourceMLIR, SourceMemo &memo,
                   std::string &error) {
    memo.tsan = nullptr;
    memo.plain = nullptr;
    memo.tsanModule.reset();
    memo.baselines.clear();
    memo.warnCount = 0;
    memo.tsanError.clear();
    memo.llvmCtx = std::make_unique<llvm::LLVMContext>();
    memo.sourceMLIR = sourceMLIR;
    memo.sourceHash = hashString(sourceMLIR);
    ensureCacheDirs();

    std::unique_ptr<llvm::Module> llvmModule;
    if (!loadModuleCache(memo.sourceHash, *memo.llvmCtx, llvmModule, nullptr,
                         &memo.sourceJitLLVM, nullptr, error)) {
        error.clear();
        if (!lowerAndTranslate(module, mlirCtx, *memo.llvmCtx, "source",
                               nullptr, &memo.sourceJitLLVM, llvmModule,
                               error))
            return false;
        saveModuleCache(memo.sourceHash, *llvmModule, "", memo.sourceJitLLVM);
    }
    memo.tsanModule = std::move(llvmModule);
    std::string compileError;
    memo.plain = compileLLVMModuleToFunction(
        llvm::CloneModule(*memo.tsanModule), &compileError, false,
        kLegacyJitOptLevel);
    if (!memo.plain) {
        error = "JIT compile error (original): " + compileError;
        return false;
    }
    return true;
}

// returns the TSan-instrumented source binary, compiling it on first use and
// keeping it alive for later runs of the same file
std::function<std::vector<int64_t>()> sourceTsanBinary(SourceMemo &memo,
                                                       std::string &error) {
    if (memo.tsan)
        return memo.tsan;
    if (!memo.tsanModule) {
        error = memo.tsanError;
        return nullptr;
    }
    std::string compileError;
    memo.tsan = compileLLVMModuleToFunction(std::move(memo.tsanModule),
                                            &compileError, true,
                                            kLegacyJitOptLevel);
    if (!memo.tsan)
        memo.tsanError = "JIT compile error (original): " + compileError;
    if (memo.tsan)
        return memo.tsan;
    error = memo.tsanError;
    return nullptr;
}

// the static destructor for gSourceMemo would otherwise run after TSan
// teardown begins and crash in JIT teardown
void shutdownPipeline() {
    gSourceMemo.clear();
}

} // namespace mlir_mracle
