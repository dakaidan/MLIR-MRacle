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
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>
#include <cmath>

namespace mlir_mr {

// per-campaign log folder; the checkpoint file lives inside it
static std::string gCampaignDir;
static std::string gCheckpointPath;

// the 1-thread group is a determinism check, so a small budget suffices
static constexpr int kDeterminismReps = 32;

// per-thread-level progress of one run
struct ThreadCheckpoint {
    std::string status = "pending"; // pending | done
    bool ok = true;                 // final verdict for done levels
    bool warn = false;
    std::string message;
    int originalRuns = 0;           // final run counts for done levels
    int transformedRuns = 0;
    OutcomeSetResult outcomeSet;
};

// progress of one metamorphic run, keyed by "run<i>_seed<s>" in the
// checkpoint file so interrupted campaigns resume exactly where they stopped
struct RunCheckpoint {
    std::string file;
    int seed = 0;
    std::vector<std::string> requestedTransforms;
    std::vector<std::string> appliedTransformNames;
    std::vector<std::string> appliedTransformTargets;
    bool transformApplied = false;
    std::string error;
    std::string warn;
    std::map<int, ThreadCheckpoint> threads;
};

static std::map<std::string, RunCheckpoint> gCheckpoints;

// deletes the campaign checkpoint once the pipeline finishes, so a completed
// campaign never leaves a resumable state behind
struct CheckpointCleanup {
    ~CheckpointCleanup() {
        std::error_code ec;
        std::filesystem::remove(gCheckpointPath, ec);
        std::filesystem::remove(gCheckpointPath + ".tmp", ec);
    }
};

// --- checkpoint persistence -------------------------------------------------

static JsonValue threadToJson(const ThreadCheckpoint &tc) {
    JsonValue o = jsonObject();
    jsonPut(o, "status", jsonString(tc.status));
    if (tc.status == "done") {
        jsonPut(o, "ok", jsonBool(tc.ok));
        jsonPut(o, "warn", jsonBool(tc.warn));
        jsonPut(o, "message", jsonString(tc.message));
        jsonPut(o, "original_runs", jsonInt(tc.originalRuns));
        jsonPut(o, "transformed_runs", jsonInt(tc.transformedRuns));
        jsonPut(o, "outcome_set", outcomeSetResultToJson(tc.outcomeSet));
    }
    return o;
}

static void threadFromJson(const llvm::json::Object *o, ThreadCheckpoint &tc) {
    if (!o)
        return;
    if (auto s = o->getString("status"))
        tc.status = s->str();
    if (tc.status == "done") {
        if (auto b = o->getBoolean("ok"))
            tc.ok = *b;
        if (auto b = o->getBoolean("warn"))
            tc.warn = *b;
        if (auto s = o->getString("message"))
            tc.message = s->str();
        if (auto s = o->getInteger("original_runs"))
            tc.originalRuns = static_cast<int>(*s);
        if (auto s = o->getInteger("transformed_runs"))
            tc.transformedRuns = static_cast<int>(*s);
        if (const auto *os = o->getObject("outcome_set")) {
            OutcomeSetResult result;
            if (outcomeSetResultFromJson(*os, result)) {
                result.source.totalRuns = tc.originalRuns;
                result.transformed.totalRuns = tc.transformedRuns;
                tc.outcomeSet = std::move(result);
            }
        }
    }
}

static void saveCheckpoints() {
    JsonValue root = jsonObject();
    jsonPut(root, "oracle_version", jsonInt(5));
    JsonValue runs = jsonObject();
    for (const auto &[key, rc] : gCheckpoints) {
        JsonValue r = jsonObject();
        jsonPut(r, "file", jsonString(rc.file));
        jsonPut(r, "seed", jsonInt(rc.seed));
        jsonPut(r, "transform_applied", jsonBool(rc.transformApplied));
        jsonPut(r, "error", jsonString(rc.error));
        jsonPut(r, "warn", jsonString(rc.warn));
        JsonValue requested = jsonArray();
        for (const auto &t : rc.requestedTransforms)
            jsonPush(requested, jsonString(t));
        jsonPut(r, "requested_transforms", std::move(requested));
        JsonValue applied = jsonArray();
        for (size_t i = 0; i < rc.appliedTransformNames.size(); ++i) {
            JsonValue a = jsonObject();
            jsonPut(a, "name", jsonString(rc.appliedTransformNames[i]));
            jsonPut(a, "target_function",
                    jsonString(i < rc.appliedTransformTargets.size()
                                   ? rc.appliedTransformTargets[i]
                                   : std::string()));
            jsonPush(applied, std::move(a));
        }
        jsonPut(r, "applied_transforms", std::move(applied));
        JsonValue threads = jsonObject();
        for (const auto &[t, tc] : rc.threads)
            jsonPut(threads, std::to_string(t), threadToJson(tc));
        jsonPut(r, "threads", std::move(threads));
        jsonPut(runs, key, std::move(r));
    }
    jsonPut(root, "runs", std::move(runs));

    std::string buf;
    llvm::raw_string_ostream os(buf);
    printJson(root, os);
    os << "\n";
    os.flush();

    std::string tmp = gCheckpointPath + ".tmp";
    {
        std::ofstream f(tmp);
        f << buf;
    }
    std::error_code ec;
    std::filesystem::rename(tmp, gCheckpointPath, ec);
}

static void loadCheckpoints() {
    std::ifstream f(gCheckpointPath);
    if (!f)
        return;
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    auto parsed = llvm::json::parse(content);
    if (!parsed)
        return;
    const auto *root = parsed->getAsObject();
    if (!root)
        return;
    // checkpoint schema is coupled to the oracle version; a mismatch means
    // the file predates the current result format and is discarded
    if (auto v = root->getInteger("oracle_version"); !v ||
                 *v != kResultSchemaVersion)
        return;
    const auto *runs = root->getObject("runs");
    if (!runs)
        return;
    for (const auto &kv : *runs) {
        const auto *r = kv.second.getAsObject();
        if (!r)
            continue;
        RunCheckpoint rc;
        if (auto s = r->getString("file"))
            rc.file = s->str();
        if (auto s = r->getInteger("seed"))
            rc.seed = static_cast<int>(*s);
        if (auto b = r->getBoolean("transform_applied"))
            rc.transformApplied = *b;
        if (auto s = r->getString("error"))
            rc.error = s->str();
        if (auto s = r->getString("warn"))
            rc.warn = s->str();
        if (const auto *requested = r->getArray("requested_transforms"))
            for (const auto &rv : *requested)
                if (auto s = rv.getAsString())
                    rc.requestedTransforms.push_back(s->str());
        if (const auto *applied = r->getArray("applied_transforms"))
            for (const auto &av : *applied)
                if (const auto *ao = av.getAsObject()) {
                    if (auto s = ao->getString("name"))
                        rc.appliedTransformNames.push_back(s->str());
                    if (auto s = ao->getString("target_function"))
                        rc.appliedTransformTargets.push_back(s->str());
                }
        if (const auto *threads = r->getObject("threads"))
            for (const auto &kv : *threads) {
                ThreadCheckpoint tc;
                threadFromJson(kv.second.getAsObject(), tc);
                rc.threads[std::stoi(kv.first.str())] = std::move(tc);
            }
        gCheckpoints[kv.first.str()] = std::move(rc);
    }
}

static bool allThreadsDone(const RunCheckpoint &rc) {
    if (rc.threads.empty())
        return false;
    for (const auto &[t, tc] : rc.threads)
        if (tc.status != "done")
            return false;
    return true;
}

// maps an oracle verdict onto the thread-level status string
static std::string statusFromVerdict(bool ok, bool warn) {
    if (!ok)
        return "ERROR";
    return warn ? "WARN" : "OK";
}

static ThreadGroupResult threadResultFromCheckpoint(
    int t, const ThreadCheckpoint &tc) {
    ThreadGroupResult tg;
    tg.numThreads = t;
    tg.status = statusFromVerdict(tc.ok, tc.warn);
    tg.message = tc.message;
    tg.originalRuns = tc.originalRuns;
    tg.transformedRuns = tc.transformedRuns;
    tg.outcomeSet = tc.outcomeSet;
    return tg;
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
static std::filesystem::path cacheRoot() { return "cache/v2"; }

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

// the baseline cache key includes the sampling budget so a campaign run with
// a different --reps never reuses a baseline collected with another budget
static std::filesystem::path baselineCachePath(const std::string &hash,
                                               const std::string &variant,
                                               int reps) {
    return baselineCacheDir() /
           (hash + "_r" + std::to_string(reps) + "_" + variant + ".json");
}

static void ensureCacheDirs() {
    std::error_code ec;
    std::filesystem::create_directories(moduleCacheDir(), ec);
    std::filesystem::create_directories(baselineCacheDir(), ec);
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
    ensureCacheDirs();
    saveCachedModule(moduleCachePath(hash).string(), module);
    writeFileContent(loweredCachePath(hash).string(), loweredMLIR);
    writeFileContent(llvmIRCachePath(hash).string(), llvmIR);
}

static bool loadBaselineCache(const std::string &sourceHash,
                              const std::string &variant, int reps,
                              ObservedOutcomeSet &set) {
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
    ensureCacheDirs();
    std::string buf;
    llvm::raw_string_ostream os(buf);
    printJson(observedOutcomeSetToJson(set), os);
    os << "\n";
    os.flush();
    writeFileContent(baselineCachePath(sourceHash, variant, reps).string(),
                     buf);
}

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
};

static std::map<std::string, SourceMemo> gSourceMemo; // keyed by file path

// invalidates a baseline entry (memory and disk) once the per-run retest cap
// is reached with outcomes still missing: the entry can no longer learn, so
// the next run of this file recollects from scratch and can take in new
// possible values
static void clearBaseline(SourceMemo &memo, const std::string &baseKey,
                          int reps) {
    memo.baselines.erase(baseKey);
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
                          SourceMemo &memo, std::string &error) {
    memo.tsan = nullptr;
    memo.plain = nullptr;
    memo.tsanModule.reset();
    memo.baselines.clear();
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
        llvm::CloneModule(*memo.tsanModule), &compileError, false);
    if (!memo.plain) {
        error = "JIT compile error (original): " + compileError;
        return false;
    }
    return true;
}

// returns the TSan-instrumented source binary, compiling it on first use and
// keeping it alive for later runs of the same file
static std::function<std::vector<int64_t>()> sourceTsanBinary(SourceMemo &memo,
                                                              std::string &error) {
    if (memo.tsan)
        return memo.tsan;
    if (!memo.tsanModule) {
        error = memo.tsanError;
        return nullptr;
    }
    std::string compileError;
    memo.tsan = compileLLVMModuleToFunction(std::move(memo.tsanModule),
                                            &compileError, true);
    if (!memo.tsan)
        memo.tsanError = "JIT compile error (original): " + compileError;
    if (memo.tsan)
        return memo.tsan;
    error = memo.tsanError;
    return nullptr;
}

// single run mode, returns a RunInfo struct with the results of the run.
// cp records per-thread progress into the campaign checkpoint so a killed
// invocation can be resumed without discarding executed batches.
static RunInfo runSingle(const std::string &inputFile, int seed,
                         int runIdx, const std::string &transform,
                         int maxApply, int tsanPercent, int reps,
                         int retestReps, int maxSourceReps,
                         int thresholdPct, RunCheckpoint &cp) {
    MLIRSetup setup(seed, runIdx, transform, maxApply);

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
                           setup.runInfo.error))
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

    // record what this run applied, so a skipped run can be reconstructed
    // from the checkpoint instead of being re-executed
    cp.requestedTransforms = setup.runInfo.requestedTransforms;
    cp.transformApplied = setup.runInfo.transformApplied;
    cp.appliedTransformNames.clear();
    cp.appliedTransformTargets.clear();
    for (const auto &at : setup.runInfo.appliedTransforms) {
        cp.appliedTransformNames.push_back(at.name);
        cp.appliedTransformTargets.push_back(at.targetFunction);
    }

    std::string compileError;

    // the 1-thread determinism check never uses TSan; the multi-threaded
    // comparisons use the variant chosen above. The transformed module is
    // always compiled plain so the deterministic single-thread check runs
    // first without paying for instrumentation it does not need.
    auto transfPlain = compileLLVMModuleToFunction(
        llvm::CloneModule(*moduleToTransformLLVM), &compileError, false);
    if (!transfPlain) {
        setup.runInfo.error =
            "JIT compile error (transformed): " + compileError;
        return setup.runInfo;
    }

    std::function<std::vector<int64_t>()> origTsan, transfTsan;
    if (useTsan) {
        origTsan = sourceTsanBinary(memo, compileError);
        if (!origTsan) {
            setup.runInfo.error = compileError;
            return setup.runInfo;
        }
        transfTsan = compileLLVMModuleToFunction(
            llvm::CloneModule(*moduleToTransformLLVM), &compileError, true);
        if (!transfTsan) {
            setup.runInfo.error =
                "JIT compile error (transformed): " + compileError;
            return setup.runInfo;
        }
    }

    // compare the observed outcome sets for each thread count separately.
    // Every tuple of return values is compared as one unit, so swapped or
    // duplicated outputs stay distinguishable; the 1-thread group is a
    // determinism check (no concurrency means no TSan either). Progress is
    // checkpointed once per run.
    for (int t : kThreadCounts) {
        ThreadCheckpoint &tc = cp.threads[t];

        // a level completed in a previous invocation is replayed from the
        // checkpoint instead of re-executed
        if (tc.status == "done") {
            setup.runInfo.threadResults.push_back(
                threadResultFromCheckpoint(t, tc));
            if (!tc.ok) {
                setup.runInfo.error =
                    "threads=" + std::to_string(t) + ": " + tc.message;
                break;
            }
            if (tc.warn)
                setup.runInfo.warn = tc.message;
            continue;
        }

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
            std::to_string(t) + (tsanVariant ? ":tsan" : "");
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
                saveBaselineCache(memo.sourceHash, baseKey, sweepReps, srcSet);
            }
        }
        ObservedOutcomeSet trSet = collectOutcomeSet(transfFn, sweepReps, t);

        // retest the source when the transformed side found outcomes that are
        // missing from the baseline, up to a hard per-entry cap; any outcomes
        // found are merged back into the in-memory baseline
        if (t != 1) {
            auto missing =
                outcomeSetDifference(trSet.outcomes, srcSet.outcomes);
            while (!missing.empty() && srcRuns < maxSourceReps) {
                int extra = std::min(retestReps, maxSourceReps - srcRuns);
                ObservedOutcomeSet extraSet =
                    collectOutcomeSet(origFn, extra, t);
                srcSet = mergeOutcomeSets(srcSet, extraSet);
                srcRuns += extra;
                memo.baselines[baseKey] = srcSet;
                saveBaselineCache(memo.sourceHash, baseKey, sweepReps, srcSet);
                missing =
                    outcomeSetDifference(trSet.outcomes, srcSet.outcomes);
            }
            // the cap was reached with outcomes still missing: this entry can
            // no longer learn, so drop it and let the next run of the file
            // recollect from scratch
            if (!missing.empty())
                clearBaseline(memo, baseKey, sweepReps);
        }

        OutcomeSetResult outcomeSet =
            compareOutcomeSets(srcSet, trSet, t, thresholdPct);
        CompareResult cmp = outcomeSet.compare;

        setup.runInfo.threadResults.push_back(threadResultFromCompare(
            t, cmp, srcRuns, sweepReps, std::move(outcomeSet)));

        tc.status = "done";
        tc.ok = cmp.ok;
        tc.warn = cmp.warn;
        tc.message = cmp.message;
        tc.originalRuns = srcRuns;
        tc.transformedRuns = sweepReps;
        tc.outcomeSet = std::move(outcomeSet);

        if (!cmp.ok) {
            setup.runInfo.error =
                "threads=" + std::to_string(t) + ": " + cmp.message;
            break;
        }
        if (cmp.warn)
            setup.runInfo.warn = cmp.message;
    }

    cp.error = setup.runInfo.error;
    cp.warn = setup.runInfo.warn;
    saveCheckpoints();

    return setup.runInfo;
}

// straight execution mode: parse, lower, JIT and run the source program at
// every team size without any transformation or comparison. Joint outcome
// frequencies are recorded per thread count.
static ExecutionRunResult executeSingle(const std::string &inputFile, int seed,
                                        int runIdx, int reps) {
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
                           result.error))
            return result;
        memo = std::move(fresh);
    }
    result.llvmIR = memo.sourceJitLLVM;

    for (int t : kThreadCounts) {
        ObservedOutcomeSet set;
        int runs;
        std::string baseKey = std::to_string(t);
        
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
// records joint outcome frequencies per thread count. There is no
// checkpointing or status classification.
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
            result.campaignDir = gCampaignDir;
            return result;
        }
    } else {
        files.push_back(opts.inputFile);
    }
    for (size_t i = 0; i < files.size(); ++i) {
        int runSeed = opts.seed >= 0 ? opts.seed : 42;
        result.runs.push_back(executeSingle(
            files[i], runSeed, opts.runNumber + static_cast<int>(i),
            opts.reps));
    }
    result.campaignDir = gCampaignDir;
    return result;
}

// core pipeline function, runs the metamorphic testing pipeline for the given options
// returns a PipelineResult struct with the results of all runs
PipelineResult runPipeline(const PipelineOptions &opts) {
    PipelineResult result;
    result.runs.reserve(opts.numRuns);

    // the campaign folder is fixed by the user for resumable campaigns or
    // timestamped otherwise; the checkpoint file lives inside it
    gCheckpointPath.clear();
    gCheckpoints.clear();
    createCampaignDir(opts);
    gCheckpointPath =
        (std::filesystem::path(gCampaignDir) / "checkpoint.json").string();
    CheckpointCleanup cleanup;

    loadCheckpoints();

    // if multi mode is requested, collect all .mlir files in the given folder
    // if none are found return a RunInfo with an error
    std::vector<std::string> multiFiles;
    if (!opts.multiFolder.empty()) {
        multiFiles = collectMLIRFiles(opts.multiFolder);
        if (multiFiles.empty()) {
            RunInfo errInfo;
            errInfo.error = "no .mlir files in " + opts.multiFolder;
            errInfo.runNumber = opts.runNumber;
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

        std::string inputFile = pickInputFile(opts, multiFiles, runIdx, runSeed);

        // run single run and collect the RunInfo; errors and warnings are
        // carried inside the RunInfo and surfaced by the caller, not printed
        // here. Runs and seeds are deterministic from the options, so a run
        // whose thread levels all completed in a previous invocation is
        // reconstructed from the checkpoint instead of being executed again.
        std::string runKey = "run" + std::to_string(runIdx) + "_seed" +
                             std::to_string(runSeed);
        RunCheckpoint &cp = gCheckpoints[runKey];
        if (cp.file.empty()) {
            cp.file = inputFile;
            cp.seed = runSeed;
        }

        if (allThreadsDone(cp)) {
            RunInfo info;
            info.runNumber = runIdx;
            info.seed = runSeed;
            info.file = cp.file;
            info.requestedTransforms = cp.requestedTransforms;
            info.transformApplied = cp.transformApplied;
            for (size_t k = 0; k < cp.appliedTransformNames.size(); ++k)
                info.appliedTransforms.push_back(
                    {cp.appliedTransformNames[k],
                     k < cp.appliedTransformTargets.size()
                         ? cp.appliedTransformTargets[k]
                         : std::string()});
            for (int t : kThreadCounts) {
                auto it = cp.threads.find(t);
                if (it == cp.threads.end())
                    continue;
                info.threadResults.push_back(
                    threadResultFromCheckpoint(t, it->second));
            }
            info.error = cp.error;
            info.warn = cp.warn;
            result.runs.push_back(std::move(info));
            continue;
        }

        RunInfo info = runSingle(inputFile, runSeed, runIdx,
                                 opts.transform, opts.maxApply,
                                 opts.tsanPercent, opts.reps,
                                 opts.retestReps, opts.maxSourceReps,
                                 opts.thresholdPct, cp);

        result.runs.push_back(std::move(info));
    }

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
// No campaign checkpointing or execution state is used.
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
        result.runs.push_back(
            emitSingle(inputFile, runSeed, runIdx, opts.transform,
                       opts.maxApply));
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