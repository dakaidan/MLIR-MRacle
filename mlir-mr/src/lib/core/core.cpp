#include "mlir-mr/core/core.h"
#include "mlir-mr/backend/jit/jit.h"
#include "mlir-mr/backend/lowering/lowering.h"
#include "mlir-mr/io/io.h"
#include "mlir-mr/oracle/oracle.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace mlir_mr {

// per-campaign log folder; the checkpoint file lives inside it
static std::string gCampaignDir;
static std::string gCheckpointPath;

// per-thread-level progress of one run
struct ThreadCheckpoint {
    std::string status = "pending"; // pending | running | done
    int srcRuns = 0;                // executions reflected in srcCounts
    int trRuns = 0;
    OutcomeCounts srcCounts;
    OutcomeCounts trCounts;
    bool ok = true;                 // final verdict for done levels
    bool warn = false;
    std::string message;
    int originalRuns = 0;           // final run counts for done levels
    int transformedRuns = 0;
    std::vector<FisherResult> variables;
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

// --- checkpoint persistence -------------------------------------------------

static llvm::json::Array countsToJson(const OutcomeCounts &counts) {
    llvm::json::Array arr;
    for (const auto &m : counts) {
        llvm::json::Array col;
        for (const auto &[val, c] : m) {
            llvm::json::Array pair;
            pair.push_back(llvm::json::Value(val));
            pair.push_back(llvm::json::Value(static_cast<int64_t>(c)));
            col.push_back(std::move(pair));
        }
        arr.push_back(std::move(col));
    }
    return arr;
}

static OutcomeCounts countsFromJson(const llvm::json::Array *arr) {
    OutcomeCounts counts;
    if (!arr)
        return counts;
    counts.reserve(arr->size());
    for (const auto &colVal : *arr) {
        std::map<int64_t, int> m;
        if (const auto *col = colVal.getAsArray())
            for (const auto &pairVal : *col)
                if (const auto *pair = pairVal.getAsArray();
                    pair && pair->size() == 2)
                    if (auto v = (*pair)[0].getAsInteger())
                        if (auto c = (*pair)[1].getAsInteger())
                            m[static_cast<int64_t>(*v)] =
                                static_cast<int>(*c);
        counts.push_back(std::move(m));
    }
    return counts;
}

static llvm::json::Object threadToJson(const ThreadCheckpoint &tc) {
    llvm::json::Object o;
    o["status"] = tc.status;
    o["src_runs"] = static_cast<int64_t>(tc.srcRuns);
    o["tr_runs"] = static_cast<int64_t>(tc.trRuns);
    if (tc.status == "running") {
        o["src_counts"] = countsToJson(tc.srcCounts);
        o["tr_counts"] = countsToJson(tc.trCounts);
    } else if (tc.status == "done") {
        o["ok"] = tc.ok;
        o["warn"] = tc.warn;
        o["message"] = tc.message;
        o["original_runs"] = static_cast<int64_t>(tc.originalRuns);
        o["transformed_runs"] = static_cast<int64_t>(tc.transformedRuns);
        llvm::json::Array vars;
        for (const auto &fr : tc.variables)
            vars.push_back(fisherResultToJson(fr));
        o["variables"] = std::move(vars);
    }
    return o;
}

static void threadFromJson(const llvm::json::Object *o, ThreadCheckpoint &tc) {
    if (!o)
        return;
    if (auto s = o->getString("status"))
        tc.status = s->str();
    if (auto s = o->getInteger("src_runs"))
        tc.srcRuns = static_cast<int>(*s);
    if (auto s = o->getInteger("tr_runs"))
        tc.trRuns = static_cast<int>(*s);
    if (tc.status == "running") {
        tc.srcCounts = countsFromJson(o->getArray("src_counts"));
        tc.trCounts = countsFromJson(o->getArray("tr_counts"));
    } else if (tc.status == "done") {
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
        if (const auto *vars = o->getArray("variables"))
            for (const auto &vv : *vars)
                if (const auto *vo = vv.getAsObject()) {
                    FisherResult fr;
                    if (fisherResultFromJson(*vo, fr))
                        tc.variables.push_back(std::move(fr));
                }
    }
}

static void saveCheckpoints() {
    llvm::json::Object root;
    llvm::json::Object runs;
    for (const auto &[key, rc] : gCheckpoints) {
        llvm::json::Object r;
        r["file"] = rc.file;
        r["seed"] = static_cast<int64_t>(rc.seed);
        r["transform_applied"] = rc.transformApplied;
        r["error"] = rc.error;
        r["warn"] = rc.warn;
        llvm::json::Array requested;
        for (const auto &t : rc.requestedTransforms)
            requested.push_back(t);
        r["requested_transforms"] = std::move(requested);
        llvm::json::Array applied;
        for (size_t i = 0; i < rc.appliedTransformNames.size(); ++i) {
            llvm::json::Object a;
            a["name"] = rc.appliedTransformNames[i];
            a["target_function"] =
                i < rc.appliedTransformTargets.size()
                    ? rc.appliedTransformTargets[i]
                    : std::string();
            applied.push_back(std::move(a));
        }
        r["applied_transforms"] = std::move(applied);
        llvm::json::Object threads;
        for (const auto &[t, tc] : rc.threads)
            threads[std::to_string(t)] = threadToJson(tc);
        r["threads"] = std::move(threads);
        runs[key] = std::move(r);
    }
    root["runs"] = std::move(runs);

    std::string buf;
    llvm::raw_string_ostream os(buf);
    os << llvm::json::Value(std::move(root)) << "\n";
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
    const auto *runs = root ? root->getObject("runs") : nullptr;
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

// Lazily creates the per-campaign log folder and returns its path.
static const std::string &ensureCampaignDir() {
    std::error_code ec;
    std::filesystem::create_directories(gCampaignDir, ec);
    return gCampaignDir;
}

// Writes the artifacts of a run to logs
static void saveRunArtifacts(const RunInfo &info,
                             const std::string &transformedMLIR,
                             const std::string &loweredMLIR,
                             const std::string &jitLLVM) {
    if (transformedMLIR.empty() && loweredMLIR.empty() && jitLLVM.empty())
        return;

    std::filesystem::path dir =
        std::filesystem::path(ensureCampaignDir()) /
        ("run" + std::to_string(info.runNumber) + "_seed" +
         std::to_string(info.seed));
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);

    auto writeIfNonEmpty = [&](const std::string &name,
                               const std::string &content) {
        if (content.empty())
            return;
        std::ofstream os((dir / name).string());
        os << content;
    };
    writeIfNonEmpty("transformed.mlir", transformedMLIR);
    writeIfNonEmpty("lowered.mlir", loweredMLIR);
    writeIfNonEmpty("jit.ll", jitLLVM);
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
    return files;
}

// single run mode, returns a RunInfo struct with the results of the run.
// cp records per-thread progress into the campaign checkpoint so a killed
// invocation can be resumed without discarding executed batches.
static RunInfo runSingle(const std::string &inputFile, int seed,
                         int runIdx, const std::string &transform,
                         int maxApply, bool printMLIR, bool log,
                         bool verbose, int tsanPercent, RunCheckpoint &cp) {
    MLIRSetup setup(seed, runIdx, transform, maxApply);
    setup.runInfo.file = inputFile;

    std::string transformedMLIRStr, loweredMLIRStr, jitLLVMStr;
    auto saveArtifacts = [&]() {
        saveRunArtifacts(setup.runInfo, transformedMLIRStr,
                         loweredMLIRStr, jitLLVMStr);
    };

    // set up a diagnostic handler to capture errors at various layers and store them in the RunInfo struct
    mlir::ScopedDiagnosticHandler diagHandler(
        &setup.mlirContext, [&](mlir::Diagnostic &diag) {
            if (!setup.runInfo.error.empty())
                setup.runInfo.error += "; ";
            setup.runInfo.error += diag.str();
            return mlir::success();
        });

    // attempt to parse module first, clearing the error if successful, otherwise return the RunInfo with the error
    setup.runInfo.error = "parse error";
    mlir::OwningOpRef<mlir::ModuleOp> originalModule =
        mlir::parseSourceFile<mlir::ModuleOp>(inputFile, &setup.mlirContext);
    if (!originalModule)
        return setup.runInfo;
    setup.runInfo.error.clear();

    // keep a copy of the parsed module; the copy is the one transformed by the pass pipeline
    mlir::OwningOpRef<mlir::ModuleOp> moduleToTransform(
        mlir::ModuleOp(originalModule->clone()));

    // similar to above but with the pass pipeline, if it fails return the RunInfo with the error, otherwise clear the error
    setup.runInfo.error = "pass pipeline failed";
    if (mlir::failed(setup.pm.run(*moduleToTransform)))
        return setup.runInfo;
    setup.runInfo.error.clear();

    // snapshot the transformed MLIR before lowering overwrites the module;
    // it is also the output used by --print-mlir
    transformedMLIRStr = dumpMLIR(*moduleToTransform);
    if (printMLIR)
        setup.runInfo.mlirOutput = transformedMLIRStr;

    // lower the source module to LLVM IR
    setup.runInfo.error = "lowering of source module to LLVM failed";
    if (mlir::failed(mlir_mr::lowerToLLVM(*originalModule, &setup.mlirContext))) {
        saveArtifacts();
        return setup.runInfo;
    }
    setup.runInfo.error.clear();

    setup.runInfo.error = "translation of source module to LLVM IR failed";
    std::unique_ptr<llvm::Module> originalModuleLLVM =
        mlir::translateModuleToLLVMIR(*originalModule, setup.llvmContext);
    if (!originalModuleLLVM) {
        saveArtifacts();
        return setup.runInfo;
    }
    setup.runInfo.error.clear();

    // lower the transformed module to LLVM IR
    setup.runInfo.error = "lowering of transformed module to LLVM failed";
    if (mlir::failed(mlir_mr::lowerToLLVM(*moduleToTransform, &setup.mlirContext))) {
        saveArtifacts();
        return setup.runInfo;
    }
    setup.runInfo.error.clear();

    // snapshot the lowered MLIR (LLVM dialect only)
    loweredMLIRStr = dumpMLIR(*moduleToTransform);

    setup.runInfo.error = "translation of transformed module to LLVM IR failed";
    std::unique_ptr<llvm::Module> moduleToTransformLLVM =
        mlir::translateModuleToLLVMIR(*moduleToTransform, setup.llvmContext);
    if (!moduleToTransformLLVM) {
        saveArtifacts();
        return setup.runInfo;
    }
    setup.runInfo.error.clear();

    // snapshot the JIT-ready LLVM IR
    jitLLVMStr = dumpLLVM(*moduleToTransformLLVM);

    // TSan instruments memory accesses and perturbs scheduling, surfacing
    // rare outcomes; applying it to a share of runs keeps that probing
    // without paying the instrumentation cost on every run. The decision is
    // made per run (seeded, so reproducible) and applies to both modules so
    // the comparison is never between differently-instrumented code.
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
    // comparisons use the variant chosen above. Both modules are always
    // compiled plain so the deterministic single-thread check runs first
    // without paying for instrumentation it does not need.
    auto origPlain = compileLLVMModuleToFunction(
        llvm::CloneModule(*originalModuleLLVM), &compileError, false);
    if (!origPlain) {
        setup.runInfo.error =
            "JIT compile error (original): " + compileError;
        saveArtifacts();
        return setup.runInfo;
    }
    auto transfPlain = compileLLVMModuleToFunction(
        llvm::CloneModule(*moduleToTransformLLVM), &compileError, false);
    if (!transfPlain) {
        setup.runInfo.error =
            "JIT compile error (transformed): " + compileError;
        saveArtifacts();
        return setup.runInfo;
    }

    std::function<std::vector<int64_t>()> origTsan, transfTsan;
    if (useTsan) {
        origTsan = compileLLVMModuleToFunction(
            llvm::CloneModule(*originalModuleLLVM), &compileError, true);
        if (!origTsan) {
            setup.runInfo.error =
                "JIT compile error (original): " + compileError;
            saveArtifacts();
            return setup.runInfo;
        }
        transfTsan = compileLLVMModuleToFunction(
            llvm::CloneModule(*moduleToTransformLLVM), &compileError, true);
        if (!transfTsan) {
            setup.runInfo.error =
                "JIT compile error (transformed): " + compileError;
            saveArtifacts();
            return setup.runInfo;
        }
    }

    // compare outcomes for each thread count separately. The 1-thread group
    // is a determinism check (no concurrency means no TSan either); the
    // multi-threaded groups run both programs in lockstep batches with a
    // Fisher exact test per output variable, stopping early on confident
    // pass/fail verdicts. Progress is checkpointed after every batch.
    for (int t : kThreadCounts) {
        ThreadCheckpoint &tc = cp.threads[t];

        // a level completed in a previous invocation is replayed from the
        // checkpoint instead of re-executed
        if (tc.status == "done") {
            setup.runInfo.threadResults.push_back(
                {t, {tc.ok, tc.warn, tc.message}, tc.originalRuns,
                 tc.transformedRuns, tc.variables});
            if (!tc.ok) {
                setup.runInfo.error =
                    "threads=" + std::to_string(t) + ": " + tc.message;
                break;
            }
            if (tc.warn)
                setup.runInfo.warn = tc.message;
            continue;
        }

        if (t == 1) {
            OutcomeCounts origCounts =
                executeCompiled(origPlain, kRunsSingle, 1);
            OutcomeCounts transfCounts =
                executeCompiled(transfPlain, kRunsSingle, 1);
            CompareResult cmp =
                compareSingleThread(origCounts, transfCounts, kRunsSingle);
            setup.runInfo.threadResults.push_back(
                {t, cmp, kRunsSingle, kRunsSingle, {}});
            tc.status = "done";
            tc.ok = cmp.ok;
            tc.warn = cmp.warn;
            tc.message = cmp.message;
            tc.originalRuns = kRunsSingle;
            tc.transformedRuns = kRunsSingle;
            tc.variables.clear();
            saveCheckpoints();
            if (!cmp.ok) {
                setup.runInfo.error = "threads=1: " + cmp.message;
                break;
            }
            if (cmp.warn)
                setup.runInfo.warn = cmp.message;
            continue;
        }

        int maxRuns = t == 2 ? kRunsPrimary : kRunsSecondary;
        int batchSize = t == 2 ? kBatchPrimary : kBatchSecondary;
        const auto &origFn = useTsan ? origTsan : origPlain;
        const auto &transfFn = useTsan ? transfTsan : transfPlain;

        // resume from checkpointed counts when the previous invocation was
        // interrupted mid-level; otherwise start fresh
        OutcomeCounts resumeSrc =
            tc.status == "running" ? tc.srcCounts : OutcomeCounts{};
        OutcomeCounts resumeTr =
            tc.status == "running" ? tc.trCounts : OutcomeCounts{};
        tc.status = "running";

        SequentialResult seq = compareSequential(
            origFn, transfFn, t, maxRuns, batchSize, resumeSrc, resumeTr,
            [&](const OutcomeCounts &src, const OutcomeCounts &tr,
                int srcRuns, int trRuns) {
                tc.srcCounts = src;
                tc.trCounts = tr;
                tc.srcRuns = srcRuns;
                tc.trRuns = trRuns;
                saveCheckpoints();
            });

        CompareResult cmp = sequentialToCompareResult(seq, t, verbose);
        setup.runInfo.threadResults.push_back(
            {t, cmp, seq.sourceRuns, seq.transformedRuns, seq.variables});

        tc.status = "done";
        tc.ok = cmp.ok;
        tc.warn = cmp.warn;
        tc.message = cmp.message;
        tc.originalRuns = seq.sourceRuns;
        tc.transformedRuns = seq.transformedRuns;
        tc.variables = seq.variables;
        tc.srcCounts.clear();
        tc.trCounts.clear();
        saveCheckpoints();

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

    // failures are always logged; successful runs only with --log
    if (!setup.runInfo.error.empty() || log)
        saveArtifacts();

    return setup.runInfo;
}

// core pipeline function, runs the metamorphic testing pipeline for the given options
// returns a PipelineResult struct with the results of all runs
PipelineResult runPipeline(const PipelineOptions &opts) {
    PipelineResult result;
    result.runs.reserve(opts.numRuns);

    // the campaign folder is fixed by the user for resumable campaigns or
    // timestamped otherwise; the checkpoint file lives inside it
    gCampaignDir.clear();
    gCheckpointPath.clear();
    gCheckpoints.clear();
    if (!opts.campaignDir.empty()) {
        gCampaignDir = opts.campaignDir;
    } else {
        using namespace std::chrono;
        auto millis = duration_cast<milliseconds>(
            system_clock::now().time_since_epoch()).count();
        gCampaignDir =
            (std::filesystem::path("logs") /
             ("campaign_" + std::to_string(millis))).string();
    }
    std::error_code ec;
    std::filesystem::create_directories(gCampaignDir, ec);
    gCheckpointPath =
        (std::filesystem::path(gCampaignDir) / "checkpoint.json").string();
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
    std::uniform_int_distribution<size_t> fileDist(
        0, multiFiles.empty() ? 0 : multiFiles.size() - 1);

    for (int i = 0; i < opts.numRuns; ++i) {
        int runIdx = opts.runNumber + i;
        int runSeed = (opts.seed >= 0)
                          ? opts.seed
                          : static_cast<int>(rng() & 0x7FFFFFFF);

        std::string inputFile;

        // if multi mode, pick one at random
        if (!opts.multiFolder.empty())
            inputFile = multiFiles[fileDist(rng)];
        else
            // else we can assume the input file is valid, as it was checked in main.cpp
            inputFile = opts.inputFile;

        // run single run and collect the RunInfo; errors and warnings are
        // carried inside the RunInfo and surfaced by the caller, not printed
        // here, so stdout stays machine-parseable. Runs and seeds are
        // deterministic from the options, so a run whose thread levels all
        // completed in a previous invocation is reconstructed from the
        // checkpoint instead of being executed again.
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
            for (const auto &[t, tc] : cp.threads)
                info.threadResults.push_back(
                    {t, {tc.ok, tc.warn, tc.message}, tc.originalRuns,
                     tc.transformedRuns, tc.variables});
            info.error = cp.error;
            info.warn = cp.warn;
            result.runs.push_back(std::move(info));
            continue;
        }

        RunInfo info = runSingle(inputFile, runSeed, runIdx,
                                 opts.transform, opts.maxApply,
                                 opts.printMLIR, opts.log, opts.verbose,
                                 opts.tsanPercent, cp);

        result.runs.push_back(std::move(info));
    }

    return result;
}

} // namespace mlir_mr
