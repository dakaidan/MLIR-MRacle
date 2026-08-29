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
#include <set>

namespace mlir_mracle {
namespace {

// converts a pipeline error string into a structured FAIL issue
VerdictIssue failIssue(const std::string &reason) {
    VerdictIssue issue;
    issue.severity = IssueSeverity::Fail;
    issue.reason = reason;
    return issue;
}

// summarises a single binary's outcome set for the run-level union outcome sets
BinaryOutcomeSummary toBinaryOutcomeSummary(const CompiledBinary &bin) {
    BinaryOutcomeSummary summary;
    summary.identity = bin.identity;
    summary.observed = bin.threadedTotal;
    return summary;
}

// fills runInfo with the outcome sets and run counts from the execution result
void populateRunInfoFromExecution(RunInfo &info, ExecutionResult &exec) {
    info.sourceRuns = exec.sourceTotal.totalRuns;
    info.transformedRuns = exec.transformedTotal.totalRuns;
    info.sourceOutcomes = std::move(exec.sourceTotal.outcomes);
    info.sourceCounts = std::move(exec.sourceTotal.counts);
    info.transformedOutcomes = std::move(exec.transformedTotal.outcomes);
    info.transformedCounts = std::move(exec.transformedTotal.counts);

    for (const auto &bin : exec.sourceBinaries)
        info.binaryOutcomes.push_back(toBinaryOutcomeSummary(bin));
    for (const auto &bin : exec.transformedBinaries)
        info.binaryOutcomes.push_back(toBinaryOutcomeSummary(bin));
}

// replays the compilation and execution of all binaries in rounds of extra runs
// used in pipeline when results are inconclusive
OracleResult runReplayLoop(ExecutionResult &exec, int seed, int configCount,
                           const PipelineOptions &opts,
                           OracleOptions oracleOpts,
                           const llvm::Module &sourceModule,
                           const llvm::Module &transformedModule,
                           std::string &triageError) {
    OracleResult verdict = oracleCompare(exec, oracleOpts);
    int64_t binaryCount =
        std::max<int64_t>(1, static_cast<int64_t>(exec.sourceBinaries.size()));
    int replayRound = 0;

    while ((verdict.needsRerun || verdict.compare.warn()) &&
           exec.sourceTotal.totalRuns < opts.maxSourceReps) {
        int64_t budget = opts.maxSourceReps - exec.sourceTotal.totalRuns;
        int extra = static_cast<int>(std::min<int64_t>(
            opts.retestReps, budget / binaryCount));
        if (extra <= 0)
            break;
        // each round re-agitates with a seed derived directly from the run seed
        // allowing reproducible replay of the same rare states across rounds
        rerunAllBinaries(exec, extra,
                         static_cast<uint32_t>(seed) +
                             static_cast<uint32_t>(++replayRound),
                         configCount);
        verdict = oracleCompare(exec, oracleOpts);
    }
    // if the verdict is still inconclusive after the maxSourceReps cap
    // run a final TSan triage
    if (verdict.needsRerun) {
        std::string err;
        if (!runTsanTriage(sourceModule, transformedModule, exec,
                           opts.retestReps,
                           static_cast<uint32_t>(seed) + 0x9e3779b9u,
                           configCount, &err))
            triageError = err;
    }
    oracleOpts.postReruns = true;
    return oracleCompare(exec, oracleOpts);
}

} // namespace

RunInfo runSingle(const std::string &inputFile, int seed,
                  int runIdx, const PipelineOptions &opts) {
    MLIRSetup setup(seed, runIdx, opts.transform, opts.maxApply, opts.model);

    mlir::OwningOpRef<mlir::ModuleOp> originalModule;
    mlir::OwningOpRef<mlir::ModuleOp> moduleToTransform;

    // if unable to apply the requested transforms, return a runInfo with the error and no verdict
    if (!applyTransforms(setup, inputFile, originalModule, moduleToTransform)) {
        setup.runInfo.issues.push_back(failIssue(setup.runInfo.error));
        return setup.runInfo;
    }

    // apply jitter as part of the agitation sweep, fail if unable
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

    // lower and translate both modules to LLVM IR, fail if unable
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
    
    // run the execution harness: compiles all modules, runs agitation, collects results
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

    std::string triageError;
    OracleResult verdict = runReplayLoop(exec, seed, execOpts.configCount,
                                         opts, oracleOpts,
                                         *sourceLLVM, *transformedLLVM,
                                         triageError);
    populateRunInfoFromExecution(setup.runInfo, exec);

    if (!triageError.empty()) {
        setup.runInfo.error = triageError;
        setup.runInfo.issues.push_back(failIssue(triageError));
        return setup.runInfo;
    }

    setup.runInfo.issues = verdict.compare.issues;

    if (!verdict.compare.ok())
        setup.runInfo.error = verdict.compare.message();
    else if (verdict.compare.warn())
        setup.runInfo.warn = verdict.compare.message();
    return setup.runInfo;
}

PipelineResult runPipeline(const PipelineOptions &opts) {
    PipelineResult result;
    result.runs.reserve(opts.numRuns);

    createCampaignDir(opts);
    createCampaignStatusDirs(gCampaignDir);

    // multi-file handling
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

    // main pipeline loop, calls runSingle
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
