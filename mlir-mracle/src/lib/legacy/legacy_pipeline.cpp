#include "mlir-mracle/legacy/legacy_pipeline.h"

#include "mlir-mracle/backend/jit/jit.h"
#include "mlir-mracle/execution/execution.h"
#include "mlir-mracle/io/artifacts.h"
#include "mlir-mracle/io/cache.h"
#include "mlir-mracle/legacy/legacy_oracle.h"
#include "mlir-mracle/legacy/source_memo.h"
#include "mlir-mracle/pipeline/common/pipeline_common.h"
#include "mlir-mracle/io/io.h"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace mlir_mracle {

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



// single run mode of the legacy --tsan pipeline, returns a RunInfo struct
// with the results of the run.
RunInfo runLegacySingle(const std::string &inputFile, int seed,
                         int runIdx, const std::string &transform,
                         int maxApply, int tsanPercent, int reps,
                         int retestReps, int maxSourceReps,
                         int thresholdPct) {
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

    std::string compileError;

    // the 1-thread determinism check never uses TSan; the multi-threaded
    // comparisons use the variant chosen above. The transformed module is
    // always compiled plain so the deterministic single-thread check runs
    // first without paying for instrumentation it does not need.
    auto transfPlain = compileLLVMModuleToFunction(
        llvm::CloneModule(*moduleToTransformLLVM), &compileError, false,
        kLegacyJitOptLevel);
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
            llvm::CloneModule(*moduleToTransformLLVM), &compileError, true,
            kLegacyJitOptLevel);
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
            std::to_string(t) + ":o" +
            std::to_string(kLegacyJitOptLevel) +
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

// core pipeline function for --legacy: runs the metamorphic testing pipeline
// (per-thread-count baseline cache + TSan instrumentation) for the given
// options; returns a PipelineResult struct with the results of all runs
PipelineResult runLegacyPipeline(const PipelineOptions &opts) {
    PipelineResult result;
    result.runs.reserve(opts.numRuns);

    // the campaign folder is fixed by the user or timestamped otherwise;
    // runs are added to it as they complete
    createCampaignDir(opts);
    createCampaignStatusDirs(gCampaignDir);

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

    for (int i = 0; i < opts.numRuns; ++i) {
        int runIdx = opts.runNumber + i;
        int runSeed = runSeedFor(opts, runIdx);

        std::string inputFile = pickInputFile(opts, multiFiles, runIdx, runSeed);

        // run single run and collect the RunInfo; errors and warnings are
        // carried inside the RunInfo and surfaced by the caller, not printed
        // here
        RunInfo info = runLegacySingle(inputFile, runSeed, runIdx,
                                 opts.transform, opts.maxApply,
                                 opts.tsanPercent, opts.reps,
                                 opts.retestReps, opts.maxSourceReps,
                                 opts.thresholdPct);

        // publish the run into its status folder as it completes; each run's
        // artifacts are written exactly once, so incremental output costs no
        // more than a single end-of-campaign write
        saveRunArtifacts(info, statusDirFor(info), gCampaignDir);

        result.runs.push_back(std::move(info));
    }

    JsonValue arr = jsonArray();
    for (const auto &run : result.runs)
        jsonPush(arr, runInfoToJson(run));
    writeResultJson(arr, gCampaignDir);

    result.campaignDir = gCampaignDir;
    return result;
}

} // namespace mlir_mracle
