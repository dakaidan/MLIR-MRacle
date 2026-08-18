#include "mlir-mr/core/core.h"
#include "mlir-mr/backend/jit/jit.h"
#include "mlir-mr/backend/lowering/lowering.h"
#include "mlir-mr/io/io.h"
#include "mlir-mr/oracle/oracle.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>
#include <cmath>

namespace mlir_mr {

// per-campaign log folder; runs are written into it as they complete
static std::string gCampaignDir;

// the 1-thread group is a determinism check, so a small budget suffices
static constexpr int kDeterminismReps = 32;

// the first few warns of a source file are verified before they are
// reported: the source baseline is retested (up to the per-level cap) and
// merged into the baseline, and the comparison is re-judged, so a poisoned
// baseline cannot warn before it has been checked; only a warn that survives
// the extra source data stands. Later warns are trusted as-is.
static constexpr int kBaselineWarnLimit = 5;

// maps an oracle verdict onto the thread-level status string
static std::string statusFromVerdict(bool ok, bool warn) {
    if (!ok)
        return "ERROR";
    return warn ? "WARN" : "OK";
}

static ThreadGroupResult threadResultFromCompare(
    int t, const CompareResult &cmp, int srcRuns, int trRuns,
    OutcomeSetResult outcomeSet) {
    ThreadGroupResult tg;
    tg.numThreads = t;
    tg.status = statusFromVerdict(cmp.ok, cmp.warn);
    tg.message = cmp.message;
    tg.originalRuns = srcRuns;
    tg.transformedRuns = trRuns;
    tg.outcomeSet = std::move(outcomeSet);
    return tg;
}

// applies the relation-specific oracle to a source/transformed outcome-set
// pair; used for the initial judgement and for the warn-verification re-judge
static OutcomeSetResult judgeOutcomeSets(OutcomeRelation relation,
                                         const ObservedOutcomeSet &src,
                                         const ObservedOutcomeSet &tr,
                                         int t, int thresholdPct) {
    switch (relation) {
    case OutcomeRelation::Subset:
        return compareOutcomeSetsSubset(src, tr, t, thresholdPct);
    case OutcomeRelation::Superset:
        return compareOutcomeSetsSuperset(src, tr, t, thresholdPct);
    default:
        return compareOutcomeSets(src, tr, t, thresholdPct);
    }
}

// helper function for multi mode, collects all .mlir files in the given folder
static std::vector<std::string> collectMLIRFiles(const std::string &folder) {
    std::vector<std::string> files;
    std::error_code ec;
    for (const auto &entry :
         std::filesystem::directory_iterator(folder, ec)) {
        if (entry.is_regular_file() && entry.path().extension() == ".mlir")
            files.push_back(entry.path().string());
    }
    std::sort(files.begin(), files.end());
    return files;
}

// picks a random .mlir file for a run; the per-run file RNG is derived only
// from seed and run index, so a fixed seed picks the same file every time
static std::string pickInputFile(const PipelineOptions &opts,
                                 const std::vector<std::string> &multiFiles,
                                 int runIdx, int runSeed) {
    if (multiFiles.empty())
        return opts.inputFile;
    uint32_t fileSeed = static_cast<uint32_t>(runSeed) +
                        static_cast<uint32_t>(runIdx) * 0x9e3779b9u;
    std::mt19937 fileRng(fileSeed);
    std::uniform_int_distribution<size_t> dist(0, multiFiles.size() - 1);
    return multiFiles[dist(fileRng)];
}

// creates (or reuses) the campaign folder used by both pipelines
static void createCampaignDir(const PipelineOptions &opts) {
    if (!opts.campaignDir.empty()) {
        gCampaignDir = opts.campaignDir;
    } else {
        using namespace std::chrono;
        auto millis = duration_cast<milliseconds>(
            system_clock::now().time_since_epoch()).count();
        gCampaignDir =
            (std::filesystem::path("results") /
             ("campaign_" + std::to_string(millis))).string();
    }
    std::error_code ec;
    std::filesystem::create_directories(gCampaignDir, ec);
}

// writes the artifacts of a single run under
// <campaignDir>/<status>/run<N>_seed<S>/ so a long campaign publishes each
// run's output as it completes instead of buffering everything until the end
static void saveRunArtifacts(const RunInfo &run, const std::string &status,
                             const std::string &campaignDir) {
    std::filesystem::path dir =
        std::filesystem::path(campaignDir) / status /
        ("run" + std::to_string(run.runNumber) + "_seed" +
         std::to_string(run.seed));
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);

    auto writeIfNonEmpty = [&](const std::string &name,
                               const std::string &content) {
        if (content.empty())
            return;
        std::ofstream os((dir / name).string());
        os << content;
    };
    writeIfNonEmpty("source.mlir", run.sourceMLIR);
    writeIfNonEmpty("transformed.mlir", run.transformedMLIR);
    writeIfNonEmpty("lowered.mlir", run.loweredMLIR);
    writeIfNonEmpty("transformed.ll", run.jitLLVM);
    writeIfNonEmpty("source.ll", run.sourceJitLLVM);
    writeIfNonEmpty("module.bc", run.bitcode);

    std::string infoBuf;
    llvm::raw_string_ostream infoOs(infoBuf);
    printJson(runInfoToStatusJson(run), infoOs);
    infoOs << "\n";
    infoOs.flush();
    std::ofstream infoFile((dir / "run_info.json").string());
    infoFile << infoBuf;
}

// writes the .ll artifact of one execution-mode run under
// <campaignDir>/run<N>_seed<S>/
static void saveExecutionArtifacts(const ExecutionRunResult &run,
                                   const std::string &campaignDir) {
    if (run.llvmIR.empty())
        return;
    std::filesystem::path dir =
        std::filesystem::path(campaignDir) /
        ("run" + std::to_string(run.runNumber) + "_seed" +
         std::to_string(run.seed));
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    std::ofstream os((dir / "source.ll").string());
    os << run.llvmIR;
}

// writes the campaign's result.json
static void writeResultJson(const JsonValue &arr,
                            const std::string &campaignDir) {
    std::filesystem::path logPath =
        std::filesystem::path(campaignDir) / "result.json";
    std::string logBuf;
    llvm::raw_string_ostream logOs(logBuf);
    printJson(arr, logOs);
    logOs << "\n";
    logOs.flush();
    std::ofstream logFile(logPath.string());
    logFile << logBuf;
}

// parses a source file, capturing diagnostics in error
static bool parseModuleFile(const std::string &file, mlir::MLIRContext &ctx,
                            mlir::OwningOpRef<mlir::ModuleOp> &module,
                            std::string &error) {
    mlir::ScopedDiagnosticHandler diagHandler(
        &ctx, [&](mlir::Diagnostic &diag) {
            if (!error.empty())
                error += "; ";
            error += diag.str();
            return mlir::success();
        });
    error = "parse error";
    module = mlir::parseSourceFile<mlir::ModuleOp>(file, &ctx);
    if (!module)
        return false;
    error.clear();
    return true;
}

// parses a source module, clones it, and runs the metamorphic pass pipeline
// on the clone; used by both the full pipeline and emit mode
static bool applyTransforms(MLIRSetup &setup, const std::string &inputFile,
                            mlir::OwningOpRef<mlir::ModuleOp> &originalModule,
                            mlir::OwningOpRef<mlir::ModuleOp> &transformedModule) {
    setup.runInfo.file = inputFile;
    if (!parseModuleFile(inputFile, setup.mlirContext, originalModule,
                         setup.runInfo.error))
        return false;
    setup.runInfo.sourceMLIR = dumpMLIR(*originalModule);
    transformedModule = mlir::OwningOpRef<mlir::ModuleOp>(
        mlir::ModuleOp(originalModule->clone()));

    mlir::ScopedDiagnosticHandler diagHandler(
        &setup.mlirContext, [&](mlir::Diagnostic &diag) {
            if (!setup.runInfo.error.empty())
                setup.runInfo.error += "; ";
            setup.runInfo.error += diag.str();
            return mlir::success();
        });
    setup.runInfo.error = "pass pipeline failed";
    if (mlir::failed(setup.pm.run(*transformedModule)))
        return false;
    setup.runInfo.error.clear();
    setup.runInfo.transformedMLIR = dumpMLIR(*transformedModule);
    return true;
}

// lowers a module to the LLVM dialect and translates it to LLVM IR,
// capturing diagnostics in error; IR dumps are only produced when requested
static bool lowerAndTranslate(mlir::ModuleOp module,
                              mlir::MLIRContext &mlirCtx,
                              llvm::LLVMContext &llvmCtx,
                              const std::string &label,
                              std::string *loweredMLIR, std::string *llvmIR,
                              std::unique_ptr<llvm::Module> &llvmModule,
                              std::string &error) {
    mlir::ScopedDiagnosticHandler diagHandler(
        &mlirCtx, [&](mlir::Diagnostic &diag) {
            if (!error.empty())
                error += "; ";
            error += diag.str();
            return mlir::success();
        });
    error = "lowering of " + label + " module to LLVM failed";
    if (mlir::failed(mlir_mr::lowerToLLVM(module, &mlirCtx)))
        return false;
    error.clear();
    if (loweredMLIR)
        *loweredMLIR = dumpMLIR(module);
    error = "translation of " + label + " module to LLVM IR failed";
    llvmModule = mlir::translateModuleToLLVMIR(module, llvmCtx);
    if (!llvmModule)
        return false;
    error.clear();
    if (llvmIR)
        *llvmIR = dumpLLVM(*llvmModule);
    return true;
}

// persistent on-disk cache under cache/v2; lowered LLVM modules are
// cached as bitcode so later campaigns skip MLIR lowering and translation,
// and source baseline outcome sets are cached as JSON so later campaigns skip
// the initial source batch. The root is versioned so a format change in the
// payloads or the cache keys never silently reuses stale entries.
// An explicitly empty MLIR_MR_CACHE_DIR disables the disk cache entirely;
// an unset variable uses the default cache/v2 root.
static bool cachingEnabled() {
    if (const char *dir = std::getenv("MLIR_MR_CACHE_DIR"))
        return *dir != '\0';
    return true;
}

static std::filesystem::path cacheRoot() {
    if (!cachingEnabled())
        return {};
    // the runner pins the root so every invocation shares the same cache
    // regardless of the working directory
    if (const char *dir = std::getenv("MLIR_MR_CACHE_DIR"))
        return dir;
    return "cache/v2";
}

static std::filesystem::path moduleCacheDir() {
    return cacheRoot() / "modules";
}

static std::filesystem::path baselineCacheDir() {
    return cacheRoot() / "baselines";
}

static std::filesystem::path moduleCachePath(const std::string &hash) {
    return moduleCacheDir() / (hash + ".bc");
}

static std::filesystem::path loweredCachePath(const std::string &hash) {
    return moduleCacheDir() / (hash + ".lowered.mlir");
}

static std::filesystem::path llvmIRCachePath(const std::string &hash) {
    return moduleCacheDir() / (hash + ".ll");
}

// defined below; forward-declared for baselineCachePath
static std::string hashString(const std::string &s);

// the baseline cache key includes the sampling budget so a campaign run with
// a different --reps never reuses a baseline collected with another budget.
// The key also embeds the result schema version both in the hash input and in
// the file-name prefix, so a format change neither reuses nor orphans stale
// entries: the prefix lets ensureCacheDirs sweep them before a new campaign
// starts.
static std::filesystem::path baselineCachePath(const std::string &hash,
                                               const std::string &variant,
                                               int reps) {
    std::string versionedHash =
        hashString(hash + "|schema:" + std::to_string(kResultSchemaVersion));
    return baselineCacheDir() /
           ("v" + std::to_string(kResultSchemaVersion) + "_" + versionedHash +
            "_r" + std::to_string(reps) + "_" + variant + ".json");
}

// removes baseline cache files left by an older result schema: the current
// schema prefixes its baseline files with "v<N>_", so any other .json in the
// baseline cache is stale. Runs once per process, before the first baseline
// lookup or write.
static void sweepStaleBaselines() {
    if (!cachingEnabled())
        return;
    static bool swept = false;
    if (swept)
        return;
    swept = true;
    const std::string prefix =
        "v" + std::to_string(kResultSchemaVersion) + "_";
    std::error_code ec;
    for (const auto &entry :
         std::filesystem::directory_iterator(baselineCacheDir(), ec)) {
        if (entry.is_regular_file() &&
            entry.path().extension() == ".json" &&
            entry.path().filename().string().rfind(prefix, 0) != 0)
            std::filesystem::remove(entry.path(), ec);
    }
}

static void ensureCacheDirs() {
    if (!cachingEnabled())
        return;
    std::error_code ec;
    std::filesystem::create_directories(moduleCacheDir(), ec);
    std::filesystem::create_directories(baselineCacheDir(), ec);
    sweepStaleBaselines();
}

static bool readFileContent(const std::string &path, std::string &content) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;
    content.assign(std::istreambuf_iterator<char>(f),
                   std::istreambuf_iterator<char>());
    return true;
}

static bool writeFileContent(const std::string &path,
                             const std::string &content) {
    std::ofstream f(path, std::ios::binary);
    if (!f)
        return false;
    f << content;
    return true;
}

// stable 64-bit FNV-1a hash rendered as hex; cache keys only need to match
// within a machine, not be cryptographic
static std::string hashString(const std::string &s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ull;
    }
    std::string hex;
    for (int shift = 60; shift >= 0; shift -= 4)
        hex += "0123456789abcdef"[(h >> shift) & 0xF];
    return hex;
}

static bool loadCachedModule(const std::string &path, llvm::LLVMContext &ctx,
                             std::unique_ptr<llvm::Module> &module,
                             std::string &error) {
    auto bufOrErr = llvm::MemoryBuffer::getFile(path);
    if (!bufOrErr)
        return false;
    auto modOrErr = llvm::parseBitcodeFile((*bufOrErr)->getMemBufferRef(), ctx);
    if (!modOrErr) {
        error = llvm::toString(modOrErr.takeError());
        return false;
    }
    module = std::move(*modOrErr);
    return true;
}

static void saveCachedModule(const std::string &path, llvm::Module &module) {
    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec);
    if (ec)
        return;
    llvm::WriteBitcodeToFile(module, os);
}

// loads a cached module and its artifact sidecars for the given hash; returns
// false when no cache entry exists
static bool loadModuleCache(const std::string &hash, llvm::LLVMContext &ctx,
                            std::unique_ptr<llvm::Module> &module,
                            std::string *loweredMLIR, std::string *llvmIR,
                            std::string *bitcode, std::string &error) {
    if (!cachingEnabled())
        return false;
    if (!loadCachedModule(moduleCachePath(hash).string(), ctx, module, error))
        return false;
    if (loweredMLIR)
        readFileContent(loweredCachePath(hash).string(), *loweredMLIR);
    if (llvmIR)
        readFileContent(llvmIRCachePath(hash).string(), *llvmIR);
    if (bitcode)
        readFileContent(moduleCachePath(hash).string(), *bitcode);
    return true;
}

static void saveModuleCache(const std::string &hash, llvm::Module &module,
                            const std::string &loweredMLIR,
                            const std::string &llvmIR) {
    if (!cachingEnabled())
        return;
    ensureCacheDirs();
    saveCachedModule(moduleCachePath(hash).string(), module);
    writeFileContent(loweredCachePath(hash).string(), loweredMLIR);
    writeFileContent(llvmIRCachePath(hash).string(), llvmIR);
}

static bool loadBaselineCache(const std::string &sourceHash,
                              const std::string &variant, int reps,
                              ObservedOutcomeSet &set) {
    if (!cachingEnabled())
        return false;
    std::string content;
    if (!readFileContent(baselineCachePath(sourceHash, variant, reps).string(),
                         content))
        return false;
    auto parsed = llvm::json::parse(content);
    if (!parsed)
        return false;
    const auto *obj = parsed->getAsObject();
    if (!obj)
        return false;
    return observedOutcomeSetFromJson(*obj, set);
}

static void saveBaselineCache(const std::string &sourceHash,
                              const std::string &variant, int reps,
                              const ObservedOutcomeSet &set) {
    if (!cachingEnabled())
        return;
    ensureCacheDirs();
    std::string buf;
    llvm::raw_string_ostream os(buf);
    printJson(observedOutcomeSetToJson(set), os);
    os << "\n";
    os.flush();
    writeFileContent(baselineCachePath(sourceHash, variant, reps).string(),
                     buf);
}

// a baseline cache write deferred until the run verdict is known: normally
// only OK runs commit, but a run that verified a warn (extra source data
// merged into the baseline) commits as well, so the grown baseline is
// persisted whether the warn resolved or was confirmed
struct PendingBaseline {
    std::string sourceHash;
    std::string baseKey;
    int reps = 0;
    ObservedOutcomeSet set;
};

// process-lifetime memoisation of source baselines: the compiled source
// binaries and the outcome sets collected from them are kept alive for the
// whole pipeline loop, so repeated runs of the same file reuse them instead
// of recompiling or re-executing. Nothing is written to disk, so no state
// survives between processes.
struct SourceMemo {
    std::string sourceMLIR;
    std::string sourceHash;
    std::string sourceJitLLVM;
    std::unique_ptr<llvm::LLVMContext> llvmCtx;
    std::function<std::vector<int64_t>()> plain;
    std::unique_ptr<llvm::Module> tsanModule; // consumed on first TSan compile
    std::function<std::vector<int64_t>()> tsan;
    std::string tsanError;
    std::map<std::string, ObservedOutcomeSet> baselines; // key "t[:tsan]"
    int warnCount = 0; // warns of this file; the first few are verified
};

static std::map<std::string, SourceMemo> gSourceMemo; // keyed by file path

// invalidates a baseline entry (memory and disk) once the per-run retest cap
// is reached with outcomes still missing: the entry can no longer learn, so
// the next run of this file recollects from scratch and can take in new
// possible values
static void clearBaseline(SourceMemo &memo, const std::string &baseKey,
                          int reps) {
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
static bool memoizeSource(mlir::MLIRContext &mlirCtx,
                          mlir::ModuleOp module, const std::string &sourceMLIR,
                          SourceMemo &memo, std::string &error,
                          int jitOptLevel) {
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
        jitOptLevel);
    if (!memo.plain) {
        error = "JIT compile error (original): " + compileError;
        return false;
    }
    return true;
}

// returns the TSan-instrumented source binary, compiling it on first use and
// keeping it alive for later runs of the same file
static std::function<std::vector<int64_t>()> sourceTsanBinary(SourceMemo &memo,
                                                              std::string &error,
                                                              int jitOptLevel) {
    if (memo.tsan)
        return memo.tsan;
    if (!memo.tsanModule) {
        error = memo.tsanError;
        return nullptr;
    }
    std::string compileError;
    memo.tsan = compileLLVMModuleToFunction(std::move(memo.tsanModule),
                                            &compileError, true, jitOptLevel);
    if (!memo.tsan)
        memo.tsanError = "JIT compile error (original): " + compileError;
    if (memo.tsan)
        return memo.tsan;
    error = memo.tsanError;
    return nullptr;
}

// single run mode, returns a RunInfo struct with the results of the run.
static RunInfo runSingle(const std::string &inputFile, int seed,
                         int runIdx, const std::string &transform,
                         int maxApply, int tsanPercent, int reps,
                         int retestReps, int maxSourceReps,
                         int thresholdPct, int jitOptLevel) {
    MLIRSetup setup(seed, runIdx, transform, maxApply);
    std::vector<PendingBaseline> pendingBaselines;

    mlir::OwningOpRef<mlir::ModuleOp> originalModule;
    mlir::OwningOpRef<mlir::ModuleOp> moduleToTransform;

    if (!applyTransforms(setup, inputFile, originalModule, moduleToTransform))
        return setup.runInfo;

    // snapshot the transformed MLIR before lowering overwrites the module
    setup.runInfo.transformedMLIR = dumpMLIR(*moduleToTransform);

    // the transformed module is cached as bitcode keyed by its MLIR text, so
    // repeated campaigns of the same source and transforms skip lowering and
    // translation; artifacts are restored from sidecar files
    std::unique_ptr<llvm::Module> moduleToTransformLLVM;
    std::string txHash = hashString(setup.runInfo.transformedMLIR);
    if (!loadModuleCache(txHash, setup.llvmContext, moduleToTransformLLVM,
                         &setup.runInfo.loweredMLIR, &setup.runInfo.jitLLVM,
                         &setup.runInfo.bitcode, setup.runInfo.error)) {
        setup.runInfo.error.clear();
        if (!lowerAndTranslate(*moduleToTransform, setup.mlirContext,
                               setup.llvmContext, "transformed",
                               &setup.runInfo.loweredMLIR,
                               &setup.runInfo.jitLLVM, moduleToTransformLLVM,
                               setup.runInfo.error))
            return setup.runInfo;
        {
            llvm::raw_string_ostream bitcodeOs(setup.runInfo.bitcode);
            llvm::WriteBitcodeToFile(*moduleToTransformLLVM, bitcodeOs);
        }
        saveModuleCache(txHash, *moduleToTransformLLVM,
                        setup.runInfo.loweredMLIR, setup.runInfo.jitLLVM);
    }

    // reuse the source binaries kept alive for the whole pipeline; the first
    // run of a file compiles them, later runs of the same file skip it. A new
    // memo is built off to the side and only installed on success, so a
    // failed recompile leaves the previous binaries intact.
    SourceMemo &memo = gSourceMemo[inputFile];
    if (memo.sourceMLIR != setup.runInfo.sourceMLIR) {
        SourceMemo fresh;
        if (!memoizeSource(setup.mlirContext, *originalModule,
                           setup.runInfo.sourceMLIR, fresh,
                           setup.runInfo.error, jitOptLevel))
            return setup.runInfo;
        memo = std::move(fresh);
    }
    setup.runInfo.sourceJitLLVM = memo.sourceJitLLVM;

    // TSan instruments memory accesses and perturbs scheduling, surfacing
    // rare outcomes; the percentage of runs instrumented is configurable
    // (100% by default). The decision is made per run (seeded, so
    // reproducible) and applies to both modules so the comparison is never
    // between differently-instrumented code.
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> tsanDist(0, 99);
    bool useTsan = tsanDist(rng) < tsanPercent;

    std::string compileError;

    // the 1-thread determinism check never uses TSan; the multi-threaded
    // comparisons use the variant chosen above. The transformed module is
    // always compiled plain so the deterministic single-thread check runs
    // first without paying for instrumentation it does not need.
    auto transfPlain = compileLLVMModuleToFunction(
        llvm::CloneModule(*moduleToTransformLLVM), &compileError, false,
        jitOptLevel);
    if (!transfPlain) {
        setup.runInfo.error =
            "JIT compile error (transformed): " + compileError;
        return setup.runInfo;
    }

    std::function<std::vector<int64_t>()> origTsan, transfTsan;
    if (useTsan) {
        origTsan = sourceTsanBinary(memo, compileError, jitOptLevel);
        if (!origTsan) {
            setup.runInfo.error = compileError;
            return setup.runInfo;
        }
        transfTsan = compileLLVMModuleToFunction(
            llvm::CloneModule(*moduleToTransformLLVM), &compileError, true,
            jitOptLevel);
        if (!transfTsan) {
            setup.runInfo.error =
                "JIT compile error (transformed): " + compileError;
            return setup.runInfo;
        }
    }

    // compare the observed outcome sets for each thread count separately.
    // Every tuple of return values is compared as one unit, so swapped or
    // duplicated outputs stay distinguishable; the 1-thread group is a
    // determinism check (no concurrency means no TSan either).
    bool verified = false;
    for (int t : kThreadCounts) {
        // the 1-thread level only needs to confirm a single deterministic
        // outcome, so it runs on a much smaller budget than the sweep levels
        int sweepReps = (t == 1) ? kDeterminismReps : reps;

        const auto &origFn =
            t == 1 ? memo.plain : (useTsan ? origTsan : memo.plain);
        const auto &transfFn =
            t == 1 ? transfPlain : (useTsan ? transfTsan : transfPlain);

        // the source side is served from the in-memory baseline when
        // possible, then the persistent baseline cache; a full miss runs once
        // with the per-run budget and persists the result. The 1-thread level
        // is only a 32-run determinism probe, so it samples afresh every time
        // and never touches the cache.
        bool tsanVariant = t != 1 && useTsan;
        std::string baseKey =
            std::to_string(t) + ":o" + std::to_string(jitOptLevel) +
            (tsanVariant ? ":tsan" : "");
        int srcRuns;
        ObservedOutcomeSet srcSet;
        if (t == 1) {
            srcSet = collectOutcomeSet(origFn, sweepReps, t);
            srcRuns = sweepReps;
        } else {
            auto baseIt = memo.baselines.find(baseKey);
            if (baseIt != memo.baselines.end()) {
                srcSet = baseIt->second;
                srcRuns = srcSet.totalRuns;
            } else if (loadBaselineCache(memo.sourceHash, baseKey, sweepReps,
                                         srcSet)) {
                srcRuns = srcSet.totalRuns;
                memo.baselines[baseKey] = srcSet;
            } else {
                srcSet = collectOutcomeSet(origFn, sweepReps, t);
                srcRuns = sweepReps;
                memo.baselines[baseKey] = srcSet;
                pendingBaselines.push_back(
                    {memo.sourceHash, baseKey, sweepReps, srcSet});
            }
        }
        ObservedOutcomeSet trSet = collectOutcomeSet(transfFn, sweepReps, t);

        // retest until the missing-outcome side is saturated, up to a hard
        // per-level cap. For equality and subset the transformed side must be
        // contained in the source baseline, so the source is retested and
        // merged back into the baseline. For superset every source outcome
        // must survive into the transformed side, so the transformed side is
        // retested instead (it is per-run and never cached).
        if (t != 1 &&
            setup.runInfo.relation == OutcomeRelation::Superset) {
            auto missing =
                outcomeSetDifference(srcSet.outcomes, trSet.outcomes);
            while (!missing.empty() && trSet.totalRuns < maxSourceReps) {
                int extra = static_cast<int>(std::min<int64_t>(
                    retestReps, maxSourceReps - trSet.totalRuns));
                ObservedOutcomeSet extraSet =
                    collectOutcomeSet(transfFn, extra, t);
                trSet = mergeOutcomeSets(trSet, extraSet);
                missing =
                    outcomeSetDifference(srcSet.outcomes, trSet.outcomes);
            }
        } else if (t != 1) {
            auto missing =
                outcomeSetDifference(trSet.outcomes, srcSet.outcomes);
            while (!missing.empty() && srcRuns < maxSourceReps) {
                int extra = std::min(retestReps, maxSourceReps - srcRuns);
                ObservedOutcomeSet extraSet =
                    collectOutcomeSet(origFn, extra, t);
                srcSet = mergeOutcomeSets(srcSet, extraSet);
                srcRuns += extra;
                memo.baselines[baseKey] = srcSet;
                pendingBaselines.push_back(
                    {memo.sourceHash, baseKey, sweepReps, srcSet});
                missing =
                    outcomeSetDifference(trSet.outcomes, srcSet.outcomes);
            }
            // the cap was reached with outcomes still missing: this entry can
            // no longer learn, so drop it and let the next run of the file
            // recollect from scratch
            if (!missing.empty()) {
                clearBaseline(memo, baseKey, sweepReps);
                pendingBaselines.erase(
                    std::remove_if(pendingBaselines.begin(),
                                   pendingBaselines.end(),
                                   [&](const PendingBaseline &p) {
                                       return p.baseKey == baseKey &&
                                              p.reps == sweepReps;
                                   }),
                    pendingBaselines.end());
            }
        }

        OutcomeSetResult outcomeSet =
            judgeOutcomeSets(setup.runInfo.relation, srcSet, trSet, t,
                             thresholdPct);
        CompareResult cmp = outcomeSet.compare;

        // the first few warns per file are verified against extra source
        // data: the source is retested and merged into the baseline (those
        // runs stay in the baseline), and the comparison is re-judged, so a
        // poisoned baseline cannot warn before it has been checked. Only a
        // warn that survives the extra source data stands; a re-judge that
        // flips to a fail is reported as such.
        if (cmp.warn && t != 1 && memo.warnCount < kBaselineWarnLimit) {
            verified = true;
            while (cmp.warn && srcRuns < maxSourceReps) {
                int extra = std::min(retestReps, maxSourceReps - srcRuns);
                ObservedOutcomeSet extraSet =
                    collectOutcomeSet(origFn, extra, t);
                srcSet = mergeOutcomeSets(srcSet, extraSet);
                srcRuns += extra;
                memo.baselines[baseKey] = srcSet;
                pendingBaselines.push_back(
                    {memo.sourceHash, baseKey, sweepReps, srcSet});
                outcomeSet = judgeOutcomeSets(setup.runInfo.relation, srcSet,
                                              trSet, t, thresholdPct);
                cmp = outcomeSet.compare;
            }
            ++memo.warnCount;
        }

        setup.runInfo.threadResults.push_back(threadResultFromCompare(
            t, cmp, srcRuns, trSet.totalRuns, std::move(outcomeSet)));

        if (!cmp.ok) {
            setup.runInfo.error =
                "threads=" + std::to_string(t) + ": " + cmp.message;
            break;
        }
        if (cmp.warn)
            setup.runInfo.warn = cmp.message;
    }

    // commit the deferred baseline writes when the run is OK; a run that
    // verified a warn (extra source data merged into the baseline) also
    // commits, so the grown baseline is persisted whether the warn resolved
    // or was confirmed
    if (verified ||
        (setup.runInfo.error.empty() && setup.runInfo.warn.empty())) {
        for (const auto &pending : pendingBaselines)
            saveBaselineCache(pending.sourceHash, pending.baseKey, pending.reps,
                              pending.set);
    }

    return setup.runInfo;
}

// straight execution mode: parse, lower, JIT and run the source program at
// every team size without any transformation or comparison. Joint outcome
// frequencies are recorded per thread count.
static ExecutionRunResult executeSingle(const std::string &inputFile, int seed,
                                        int runIdx, int reps,
                                        int jitOptLevel) {
    ExecutionRunResult result;
    result.runNumber = runIdx;
    result.seed = seed;
    result.file = inputFile;

    mlir::MLIRContext mlirCtx;
    initializeMLIRContext(mlirCtx);

    mlir::OwningOpRef<mlir::ModuleOp> module;
    if (!parseModuleFile(inputFile, mlirCtx, module, result.error))
        return result;
    std::string sourceMLIR = dumpMLIR(*module);

    // reuse the compiled source binary and baseline outcome sets kept alive
    // for the whole pipeline; the first run of a file compiles and executes
    // the full per-thread budget, later runs of the same file reuse it. A new
    // memo is built off to the side and only installed on success.
    SourceMemo &memo = gSourceMemo[inputFile];
    if (memo.sourceMLIR != sourceMLIR) {
        SourceMemo fresh;
        if (!memoizeSource(mlirCtx, *module, sourceMLIR, fresh,
                           result.error, jitOptLevel))
            return result;
        memo = std::move(fresh);
    }
    result.llvmIR = memo.sourceJitLLVM;

    for (int t : kThreadCounts) {
        ObservedOutcomeSet set;
        int runs;
        std::string baseKey =
            std::to_string(t) + ":o" + std::to_string(jitOptLevel);
        
        auto baseIt = memo.baselines.find(baseKey);
        if (baseIt != memo.baselines.end()) {
            set = baseIt->second;
            runs = set.totalRuns;
        } else if (loadBaselineCache(memo.sourceHash, baseKey, reps, set)) {
            runs = set.totalRuns;
            memo.baselines[baseKey] = set;
        } else {
            set = collectOutcomeSet(memo.plain, reps, t);
            runs = reps;
            memo.baselines[baseKey] = set;
            saveBaselineCache(memo.sourceHash, baseKey, reps, set);
        }

        ExecutionThreadResult tr;
        tr.numThreads = t;
        tr.runs = runs;
        tr.outcomes = std::move(set.outcomes);
        tr.counts = std::move(set.counts);
        result.threadResults.push_back(std::move(tr));
    }
    return result;
}

// core pipeline function for --run: executes each input file as-is and
// records joint outcome frequencies per thread count. Each run is published
// to the campaign folder as it completes; there is no status classification.
ExecutionPipelineResult runExecutionPipeline(const PipelineOptions &opts) {
    ExecutionPipelineResult result;
    result.runs.reserve(opts.numRuns);

    createCampaignDir(opts);

    std::vector<std::string> files;
    if (!opts.multiFolder.empty()) {
        files = collectMLIRFiles(opts.multiFolder);
        if (files.empty()) {
            ExecutionRunResult errInfo;
            errInfo.error = "no .mlir files in " + opts.multiFolder;
            errInfo.runNumber = opts.runNumber;
            result.runs.push_back(std::move(errInfo));
            JsonValue errArr = jsonArray();
            jsonPush(errArr, executionRunToJson(result.runs.back()));
            writeResultJson(errArr, gCampaignDir);
            result.campaignDir = gCampaignDir;
            return result;
        }
    } else {
        files.push_back(opts.inputFile);
    }
    for (size_t i = 0; i < files.size(); ++i) {
        int runSeed = opts.seed >= 0 ? opts.seed : 42;
        ExecutionRunResult run = executeSingle(
            files[i], runSeed, opts.runNumber + static_cast<int>(i),
            opts.reps, opts.jitOptLevel);
        saveExecutionArtifacts(run, gCampaignDir);
        result.runs.push_back(std::move(run));
    }

    JsonValue arr = jsonArray();
    for (const auto &run : result.runs)
        jsonPush(arr, executionRunToJson(run));
    writeResultJson(arr, gCampaignDir);
    result.campaignDir = gCampaignDir;
    return result;
}

// core pipeline function, runs the metamorphic testing pipeline for the given options
// returns a PipelineResult struct with the results of all runs
PipelineResult runPipeline(const PipelineOptions &opts) {
    PipelineResult result;
    result.runs.reserve(opts.numRuns);

    // the campaign folder is fixed by the user or timestamped otherwise;
    // runs are added to it as they complete
    createCampaignDir(opts);
    std::error_code ec;
    for (const char *sub : {"fail", "warn", "ok"})
        std::filesystem::create_directories(
            std::filesystem::path(gCampaignDir) / sub, ec);

    // if multi mode is requested, collect all .mlir files in the given folder
    // if none are found return a RunInfo with an error
    std::vector<std::string> multiFiles;
    if (!opts.multiFolder.empty()) {
        multiFiles = collectMLIRFiles(opts.multiFolder);
        if (multiFiles.empty()) {
            RunInfo errInfo;
            errInfo.error = "no .mlir files in " + opts.multiFolder;
            errInfo.runNumber = opts.runNumber;
            saveRunArtifacts(errInfo, "fail", gCampaignDir);
            JsonValue errArr = jsonArray();
            jsonPush(errArr, runInfoToJson(errInfo));
            writeResultJson(errArr, gCampaignDir);
            result.runs.push_back(errInfo);
            result.campaignDir = gCampaignDir;
            return result;
        }
    }

    std::random_device rd;
    std::mt19937 rng(rd());

    for (int i = 0; i < opts.numRuns; ++i) {
        int runIdx = opts.runNumber + i;
        int runSeed = (opts.seed >= 0)
                          ? opts.seed
                          : static_cast<int>(rng() & 0x7FFFFFFF);

        std::string inputFile = pickInputFile(opts, multiFiles, runIdx, runSeed);

        // run single run and collect the RunInfo; errors and warnings are
        // carried inside the RunInfo and surfaced by the caller, not printed
        // here
        RunInfo info = runSingle(inputFile, runSeed, runIdx,
                                 opts.transform, opts.maxApply,
                                 opts.tsanPercent, opts.reps,
                                 opts.retestReps, opts.maxSourceReps,
                                 opts.thresholdPct, opts.jitOptLevel);

        // publish the run into its status folder as it completes; each run's
        // artifacts are written exactly once, so incremental output costs no
        // more than a single end-of-campaign write
        std::string status = runStatusString(info);
        const char *statusDir = status == "ERROR" ? "fail"
                               : status == "WARN" ? "warn" : "ok";
        saveRunArtifacts(info, statusDir, gCampaignDir);

        result.runs.push_back(std::move(info));
    }

    JsonValue arr = jsonArray();
    for (const auto &run : result.runs)
        jsonPush(arr, runInfoToJson(run));
    writeResultJson(arr, gCampaignDir);

    result.campaignDir = gCampaignDir;
    return result;
}

// emit mode single run: apply the requested transforms and return the
// resulting MLIR without lowering, JIT, or oracle comparison
static RunInfo emitSingle(const std::string &inputFile, int seed,
                          int runIdx, const std::string &transform,
                          int maxApply) {
    MLIRSetup setup(seed, runIdx, transform, maxApply);
    mlir::OwningOpRef<mlir::ModuleOp> originalModule;
    mlir::OwningOpRef<mlir::ModuleOp> moduleToTransform;
    if (!applyTransforms(setup, inputFile, originalModule, moduleToTransform))
        return setup.runInfo;
    return setup.runInfo;
}

// generator mode for --emit-mlir: applies the requested transforms to one
// file or random files from a folder and returns the transformed MLIR text.
// Each run is written to the output folder as it completes; there is no
// execution state.
PipelineResult runEmitPipeline(const PipelineOptions &opts) {
    PipelineResult result;
    result.runs.reserve(opts.numRuns);

    if (!opts.campaignDir.empty()) {
        result.campaignDir = opts.campaignDir;
    } else {
        result.campaignDir = "emitted";
    }
    std::error_code ec;
    std::filesystem::create_directories(result.campaignDir, ec);

    // generator mode overwrites its output: clear stale artifacts from a
    // previous emit so the directory only holds the current batch
    for (const auto &entry :
         std::filesystem::directory_iterator(result.campaignDir, ec)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("run", 0) == 0 && name.find("_seed") != std::string::npos)
            std::filesystem::remove(entry.path(), ec);
        else if (name == "result.json")
            std::filesystem::remove(entry.path(), ec);
    }

    std::vector<std::string> multiFiles;
    if (!opts.multiFolder.empty()) {
        multiFiles = collectMLIRFiles(opts.multiFolder);
        if (multiFiles.empty()) {
            RunInfo errInfo;
            errInfo.error = "no .mlir files in " + opts.multiFolder;
            errInfo.runNumber = opts.runNumber;
            std::string base = "run" + std::to_string(errInfo.runNumber) +
                               "_seed" + std::to_string(errInfo.seed);
            std::ofstream os(
                (std::filesystem::path(result.campaignDir) /
                 (base + ".error.txt")).string());
            os << errInfo.error;
            result.runs.push_back(errInfo);
            return result;
        }
    }

    std::random_device rd;
    std::mt19937 rng(rd());
    for (int i = 0; i < opts.numRuns; ++i) {
        int runIdx = opts.runNumber + i;
        int runSeed = (opts.seed >= 0)
                          ? opts.seed
                          : static_cast<int>(rng() & 0x7FFFFFFF);
        std::string inputFile =
            pickInputFile(opts, multiFiles, runIdx, runSeed);
        RunInfo run = emitSingle(inputFile, runSeed, runIdx, opts.transform,
                                 opts.maxApply);
        std::string base = "run" + std::to_string(run.runNumber) +
                           "_seed" + std::to_string(run.seed);
        bool ok = run.error.empty() && !run.transformedMLIR.empty();
        std::string suffix = ok ? ".mlir" : ".error.txt";
        std::ofstream os(
            (std::filesystem::path(result.campaignDir) /
             (base + suffix)).string());
        os << (ok ? run.transformedMLIR : run.error);
        result.runs.push_back(std::move(run));
    }
    return result;
}

// destroys the process-lifetime JIT state while TSan is still fully alive;
// the static destructor for gSourceMemo would otherwise run after TSan
// teardown begins and crash in JIT teardown
void shutdownCore() {
    gSourceMemo.clear();
}

} // namespace mlir_mr