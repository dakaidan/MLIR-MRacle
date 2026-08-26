#include "mlir-mracle/pipeline/pipeline.h"

#include "mlir-mracle/execution/execution.h"
#include "mlir-mracle/io/artifacts.h"
#include "mlir-mracle/io/io.h"
#include "mlir-mracle/oracle/oracle.h"
#include "mlir-mracle/pipeline/common/pipeline_common.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>
#include <cmath>

namespace mlir_mracle {
namespace {

// converts a pipeline error string into a structured FAIL issue
VerdictIssue failIssue(const std::string &reason) {
    VerdictIssue issue;
    issue.severity = IssueSeverity::Fail;
    issue.reason = reason;
    return issue;
}

// converts one per-binary execution result plus its compiled binary into the
// concise breakdown appended to run_info.json
BinaryOutcomeResult toBinaryOutcomeResult(const BinaryExecutionResult &br,
                                          const CompiledBinary &bin) {
    BinaryOutcomeResult bo;
    bo.side = br.side;
    bo.compileIndex = bin.compileIndex;
    bo.jitOptLevel = bin.jitOptLevel;
    bo.runs = static_cast<int>(br.threadedTotal.totalRuns);
    bo.outcomes = br.threadedTotal.outcomes;
    bo.counts = br.threadedTotal.counts;
    return bo;
}

// fills the run-level union outcome sets and the per-binary breakdown from
// the finished execution; the verdict data lives in these sets, so no single
// team size is reported
void populateRunInfoFromExecution(RunInfo &info, ExecutionResult &exec) {
    info.sourceRuns = exec.sourceTotal.totalRuns;
    info.transformedRuns = exec.transformedTotal.totalRuns;
    info.sourceOutcomes = std::move(exec.sourceTotal.outcomes);
    info.sourceCounts = std::move(exec.sourceTotal.counts);
    info.transformedOutcomes = std::move(exec.transformedTotal.outcomes);
    info.transformedCounts = std::move(exec.transformedTotal.counts);

    int srcCount = static_cast<int>(exec.sourceBinaries.size());
    for (size_t i = 0; i < exec.binaryResults.size(); ++i) {
        const auto &br = exec.binaryResults[i];
        const CompiledBinary &bin =
            static_cast<int>(i) < srcCount
                ? exec.sourceBinaries[i]
                : exec.transformedBinaries[i - srcCount];
        info.binaryOutcomes.push_back(toBinaryOutcomeResult(br, bin));
    }
}

// replays rare states in rounds of --reruns extra source runs per binary
// until the verdict resolves to ok/fail or the total source runs across all
// binaries reach the --max-runs cap, then returns the final post-replay
// comparison judged on merged data. When the cap is reached with rare states
// still unresolved, a final TSan-instrumented triage merges scheduling-
// perturbed runs into the totals before the final judgement.
OracleResult runReplayLoop(ExecutionResult &exec, int seed, int configCount,
                           const PipelineOptions &opts,
                           OracleOptions oracleOpts,
                           const llvm::Module &sourceModule,
                           const llvm::Module &transformedModule) {
    OracleResult verdict = oracleCompare(exec, oracleOpts);
    int64_t binaryCount =
        std::max<int64_t>(1, static_cast<int64_t>(exec.sourceBinaries.size()));
    int replayRound = 0;
    while ((verdict.needsRerun || verdict.compare.warn) &&
           exec.sourceTotal.totalRuns < opts.maxSourceReps) {
        int64_t budget = opts.maxSourceReps - exec.sourceTotal.totalRuns;
        int extra = static_cast<int>(std::min<int64_t>(
            opts.retestReps, budget / binaryCount));
        if (extra <= 0)
            break;
        // each round re-agitates with a seed derived directly from the run
        // seed (round 1 -> seed+1, ...), so the replay sequence is
        // reproducible from --seed alone and every round draws a fresh
        // team-size mix
        rerunAllBinaries(exec, extra,
                         static_cast<uint32_t>(seed) +
                             static_cast<uint32_t>(++replayRound),
                         configCount);
        verdict = oracleCompare(exec, oracleOpts);
    }
    // the cap was reached with rare states still unresolved: TSan perturbs
    // memory-access scheduling, so a final triage surfaces them or confirms
    // them absent under instrumentation; a failed triage keeps the plain
    // post-rerun verdict
    if (verdict.needsRerun)
        runTsanTriage(sourceModule, transformedModule, exec, opts.retestReps,
                      static_cast<uint32_t>(seed) + 0x9e3779b9u, configCount);
    oracleOpts.postReruns = true;
    return oracleCompare(exec, oracleOpts);
}

} // namespace

// default pipeline single run: applies the requested transforms, adds
// symmetric jitter delay chains to both modules, lowers and translates both
// modules directly (no persistent cache, no source memo), runs the agitation
// sweep through the harness, then replays rare states in rounds of --reruns
// until they resolve or the total source runs across all binaries reach the
// --max-runs cap, with a final TSan triage when the cap is reached
// unresolved. The verdict is the final post-replay comparison, judged on
// merged data.
RunInfo runSingle(const std::string &inputFile, int seed,
                  int runIdx, const PipelineOptions &opts) {
    MLIRSetup setup(seed, runIdx, opts.transform, opts.maxApply, opts.model);

    mlir::OwningOpRef<mlir::ModuleOp> originalModule;
    mlir::OwningOpRef<mlir::ModuleOp> moduleToTransform;
    if (!applyTransforms(setup, inputFile, originalModule, moduleToTransform)) {
        setup.runInfo.issues.push_back(failIssue(setup.runInfo.error));
        return setup.runInfo;
    }

    // jitter widens the race windows on both sides with the same seed, so
    // rare states surface more often without biasing the comparison
    if (!applyJitter(setup.mlirContext, *originalModule, seed,
                     setup.runInfo.error) ||
        !applyJitter(setup.mlirContext, *moduleToTransform, seed,
                     setup.runInfo.error)) {
        setup.runInfo.issues.push_back(failIssue(setup.runInfo.error));
        return setup.runInfo;
    }

    // snapshot both jittered modules before lowering overwrites them
    setup.runInfo.sourceMLIR = dumpMLIR(*originalModule);
    setup.runInfo.transformedMLIR = dumpMLIR(*moduleToTransform);

    std::unique_ptr<llvm::Module> sourceLLVM;
    if (!lowerAndTranslate(*originalModule, setup.mlirContext,
                           setup.llvmContext, "source", nullptr,
                           &setup.runInfo.sourceJitLLVM, sourceLLVM,
                           setup.runInfo.error)) {
        setup.runInfo.issues.push_back(failIssue(setup.runInfo.error));
        return setup.runInfo;
    }

    std::unique_ptr<llvm::Module> transformedLLVM;
    if (!lowerAndTranslate(*moduleToTransform, setup.mlirContext,
                           setup.llvmContext, "transformed",
                           &setup.runInfo.loweredMLIR,
                           &setup.runInfo.jitLLVM, transformedLLVM,
                           setup.runInfo.error)) {
        setup.runInfo.issues.push_back(failIssue(setup.runInfo.error));
        return setup.runInfo;
    }

    ExecutionOptions execOpts;
    execOpts.seed = static_cast<uint32_t>(seed);
    execOpts.runsPerBinary = opts.reps;
    execOpts.singleThreadRuns = kDeterminismReps;
    execOpts.compile.shuffleSeed = static_cast<uint32_t>(seed);
    ExecutionResult exec = runExecutionHarness(*sourceLLVM, *transformedLLVM,
                                               execOpts);
    if (!exec.error.empty()) {
        setup.runInfo.error = exec.error;
        setup.runInfo.issues.push_back(failIssue(setup.runInfo.error));
        return setup.runInfo;
    }

    OracleOptions oracleOpts;
    oracleOpts.relation = setup.runInfo.relation;
    oracleOpts.thresholdPct = opts.thresholdPct;

    OracleResult verdict = runReplayLoop(exec, seed, execOpts.configCount,
                                         opts, oracleOpts,
                                         *sourceLLVM, *transformedLLVM);
    populateRunInfoFromExecution(setup.runInfo, exec);

    setup.runInfo.issues = verdict.compare.issues;
    if (!verdict.compare.ok)
        setup.runInfo.error = verdict.compare.message;
    else if (verdict.compare.warn)
        setup.runInfo.warn = verdict.compare.message;
    return setup.runInfo;
}

// core pipeline function for the default agitation-sweep oracle: runs the
// oracle comparison over transformed vs source modules with rare-state
// replay. Each run is published to the campaign folder as it completes.

PipelineResult runPipeline(const PipelineOptions &opts) {
    PipelineResult result;
    result.runs.reserve(opts.numRuns);

    createCampaignDir(opts);
    createCampaignStatusDirs(gCampaignDir);

    std::vector<std::string> multiFiles;
    if (!opts.multiFolder.empty()) {
        multiFiles = collectMLIRFiles(opts.multiFolder);
        if (multiFiles.empty()) {
            RunInfo errInfo;
            errInfo.error = "no .mlir files in " + opts.multiFolder;
            errInfo.runNumber = opts.runNumber;
            saveRunArtifacts(errInfo, "fail", gCampaignDir,
                             /*unionFormat=*/true);
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
        std::string inputFile =
            pickInputFile(opts, multiFiles, runIdx, runSeed);

        RunInfo info = runSingle(inputFile, runSeed, runIdx, opts);

        saveRunArtifacts(info, statusDirFor(info), gCampaignDir,
                         /*unionFormat=*/true);
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
