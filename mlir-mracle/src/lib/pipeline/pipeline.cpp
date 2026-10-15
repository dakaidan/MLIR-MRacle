#include "mlir-mracle/pipeline/pipeline.h"

#include "mlir-mracle/execution/execution.h"
#include "mlir-mracle/io/artifacts.h"
#include "mlir-mracle/io/io.h"
#include "mlir-mracle/oracle/calibrated.h"
#include "mlir-mracle/oracle/distribution_oracle.h"
#include "mlir-mracle/oracle/oracle.h"
#include "mlir-mracle/pipeline/common/pipeline_common.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace mlir_mracle {
namespace {

// converts a pipeline error string into a structured FAIL issue
VerdictIssue failIssue(const std::string &reason) {
    VerdictIssue issue;
    issue.severity = IssueSeverity::Fail;
    issue.reason = reason;
    return issue;
}

// summarises a single campaign for the run-level union outcome sets
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

// renders a joint outcome as [a, b, ...] for the artifact record
std::string formatOutcome(const JointOutcome &outcome) {
    std::string out = "[";
    for (size_t i = 0; i < outcome.size(); ++i) {
        if (i > 0)
            out += ", ";
        out += std::to_string(outcome[i]);
    }
    out += "]";
    return out;
}

std::string distributionVerdictName(DistributionVerdict verdict) {
    switch (verdict) {
    case DistributionVerdict::Ok:
        return "OK";
    case DistributionVerdict::Fail:
        return "FAIL";
    case DistributionVerdict::Inconclusive:
        break;
    }
    return "INCONCLUSIVE";
}

// every campaign of one run draws its seed from the run seed and its index, so
// a fixed --seed reproduces the whole campaign sequence
uint32_t campaignSeedFor(int runSeed, int campaignIndex) {
    return static_cast<uint32_t>(runSeed) +
           static_cast<uint32_t>(campaignIndex) * 0x9e3779b9u;
}

// collects campaigns until the oracle can decide, then returns its verdict
// collects campaigns until the oracle can decide, then returns its verdict
bool runCampaignLoop(ExecutionResult &exec, int seed,
                     const PipelineOptions &opts, OutcomeRelation relation,
                     const llvm::Module &sourceModule,
                     const llvm::Module &transformedModule,
                     DistributionOracleResult &verdict, std::string &error,
                     CalibrationResult &calibratedVerdict) {
    CampaignOptions campaignOpts;
    campaignOpts.executionsPerCampaign = opts.executionsPerCampaign;
    campaignOpts.teamSize = opts.teamSize;
    campaignOpts.jitOptLevel = kStandardJitOptLevel;

    DistributionOracleOptions oracleOpts;
    oracleOpts.relation = relation;
    oracleOpts.maxCampaigns = static_cast<size_t>(std::max(1, opts.campaignsMax));
    oracleOpts.distribution.tvdThreshold = opts.tvdThreshold;
    oracleOpts.distribution.bootstrapReplicates = opts.bootstrapReplicates;
    oracleOpts.distribution.confidenceLevel = opts.confidenceLevel;
    oracleOpts.distribution.bootstrapSeed = static_cast<uint32_t>(seed);
    oracleOpts.distribution.maxRunsPerSide =
        static_cast<int64_t>(std::max(1, opts.campaignsMax)) *
        static_cast<int64_t>(std::max(1, opts.executionsPerCampaign));
    oracleOpts.distribution.alphaTotal = opts.alphaTotal;
    // every planned judgement counts towards the false-alarm budget, so growing
    // the sample does not turn repeated looks into findings
    oracleOpts.distribution.plannedLooks =
        std::max(1, (std::max(1, opts.campaignsMax) - 1) /
                            std::max(1, opts.campaignsInitial) +
                        1);

    CalibratedOptions calibratedOpts;
    calibratedOpts.maxCampaigns = static_cast<size_t>(std::max(1, opts.campaignsMax));
    calibratedOpts.minChange = opts.minChange;
    calibratedOpts.permutations = std::max(1, opts.bootstrapReplicates);
    calibratedOpts.seed = static_cast<uint32_t>(seed);
    calibratedOpts.alphaTotal = opts.alphaTotal;
    // every judgement the budget allows is paid for out of the same false-alarm
    // budget, so repeated looks do not turn noise into findings
    calibratedOpts.plannedLooks =
        std::max(1, (std::max(1, opts.campaignsMax) - 1) /
                            std::max(1, opts.campaignsInitial) +
                        1);

    const bool calibrated = opts.comparison == ComparisonMethod::Calibrated;

    // the loop only needs to know whether to buy more campaigns; the method
    // decides which judge produces the verdict
    auto judge = [&]() {
        if (calibrated) {
            calibratedVerdict = calibratedCompare(exec, calibratedOpts);
            verdict = DistributionOracleResult{};
            verdict.needsRerun = calibratedVerdict.needsRerun;
        } else {
            verdict = distributionOracleCompare(exec, oracleOpts);
        }
    };

    int campaignIndex = 0;
    const int initial = std::max(1, opts.campaignsInitial);
    for (; campaignIndex < initial; ++campaignIndex) {
        if (!runCampaignPair(exec, sourceModule, transformedModule,
                             campaignIndex, campaignSeedFor(seed, campaignIndex),
                             campaignOpts, error))
            return false;
    }

    judge();

    // an unresolved comparison buys another batch of independent campaigns, up
    // to the cap, so the sample grows towards the rerun budget in few steps
    while (verdict.needsRerun && campaignIndex < opts.campaignsMax) {
        const int batch = std::min(initial, opts.campaignsMax - campaignIndex);
        for (int i = 0; i < batch; ++i, ++campaignIndex) {
            if (!runCampaignPair(exec, sourceModule, transformedModule,
                                 campaignIndex,
                                 campaignSeedFor(seed, campaignIndex),
                                 campaignOpts, error))
                return false;
        }
        judge();
    }

    return true;
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

// fills the protocol and distribution record shared by both campaign judges
void fillDistributionSummary(DistributionSummary &summary,
                             const DistributionResult &distribution,
                             const PipelineOptions &opts,
                             const std::string &status) {
    summary.verdict = status == "ERROR" ? "FAIL" : status == "WARN" ? "WARN" : "OK";
    summary.tvdVerdict = distributionVerdictName(distribution.verdict);
    summary.tvd = distribution.tvd;
    summary.ciLow = distribution.ciLow;
    summary.ciHigh = distribution.ciHigh;
    summary.tvdThreshold = opts.tvdThreshold;
    summary.confidenceLevel = opts.confidenceLevel;
    summary.bootstrapReplicates = opts.bootstrapReplicates;
    summary.sourceCampaigns = static_cast<int64_t>(distribution.sourceCampaigns);
    summary.transformedCampaigns =
        static_cast<int64_t>(distribution.transformedCampaigns);
    summary.excludedCampaigns =
        static_cast<int64_t>(distribution.excludedCampaigns);
    summary.sourceRuns = distribution.sourceRuns;
    summary.transformedRuns = distribution.transformedRuns;
    summary.sourceSupport = static_cast<int64_t>(distribution.sourceSupport);
    summary.transformedSupport =
        static_cast<int64_t>(distribution.transformedSupport);
    summary.sourceEffCategories = distribution.sourceEffCategories;
    summary.transformedEffCategories = distribution.transformedEffCategories;
    summary.sourceUnseenMass = distribution.sourceUnseenMass;
    summary.transformedUnseenMass = distribution.transformedUnseenMass;
    summary.executionsPerCampaign = opts.executionsPerCampaign;
    summary.teamSize = opts.teamSize;
    summary.jitOptLevel = kStandardJitOptLevel;
}


std::string supportVerdictName(SupportVerdict verdict);

SupportViolationSummary toSupportViolationSummary(
    const SupportViolation &violation) {
    SupportViolationSummary record;
    record.outcome = formatOutcome(violation.outcome);
    record.side = violation.transformedOnly ? "transformed" : "source";
    record.count = violation.count;
    record.runs = violation.runs;
    record.rate = violation.rate;
    record.bandRate = violation.bandRate;
    record.pValue = violation.pValue;
    record.significant = violation.significant;
    record.establishable = violation.establishable;
    record.verdict = supportVerdictName(violation.verdict);
    return record;
}
std::string supportVerdictName(SupportVerdict verdict) {
    switch (verdict) {
    case SupportVerdict::Fail:
        return "FAIL";
    case SupportVerdict::Warn:
        return "WARN";
    case SupportVerdict::Candidate:
        return "CANDIDATE";
    case SupportVerdict::Ok:
        break;
    }
    return "OK";
}

// records the distributional verdict of the tvd method on the run
void applyVerdict(RunInfo &info, const DistributionOracleResult &verdict,
                  const PipelineOptions &opts) {
    info.issues = verdict.compare.issues;

    DistributionSummary summary;
    fillDistributionSummary(summary, verdict.distribution, opts,
                            runStatusString(info));
    info.distribution = std::move(summary);

    for (const auto &violation : verdict.support)
        info.supportViolations.push_back(toSupportViolationSummary(violation));
}

// records the calibrated method's verdict and the evidence behind it
void applyCalibrationVerdict(RunInfo &info, const CalibrationResult &verdict,
                             const PipelineOptions &opts) {
    info.issues = verdict.compare.issues;

    CalibrationSummary summary;
    switch (verdict.verdict) {
    case CalibrationVerdict::Fail:
        summary.verdict = "FAIL";
        break;
    case CalibrationVerdict::Inconclusive:
        summary.verdict = "INCONCLUSIVE";
        break;
    case CalibrationVerdict::Ok:
        summary.verdict = "OK";
        break;
    }
    summary.score = verdict.score;
    summary.noiseThreshold = verdict.noiseThreshold;
    summary.minChange = verdict.minChange;
    summary.decisionThreshold = verdict.decisionThreshold;
    summary.pValue = verdict.pValue;
    summary.permutations = verdict.permutations;
    summary.sourceCampaigns = static_cast<int64_t>(verdict.sourceCampaigns);
    summary.transformedCampaigns = static_cast<int64_t>(verdict.transformedCampaigns);
    summary.sourceRuns = verdict.sourceRuns;
    summary.transformedRuns = verdict.transformedRuns;
    summary.sourceSupport = static_cast<int64_t>(verdict.sourceSupport);
    summary.transformedSupport = static_cast<int64_t>(verdict.transformedSupport);
    summary.sourceOnly = static_cast<int64_t>(verdict.sourceOnly);
    summary.transformedOnly = static_cast<int64_t>(verdict.transformedOnly);
    summary.executionsPerCampaign = opts.executionsPerCampaign;
    summary.teamSize = opts.teamSize;
    summary.jitOptLevel = kStandardJitOptLevel;
    info.calibration = std::move(summary);
}

} // namespace

RunInfo runSingleAgitation(const std::string &inputFile, int seed,
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

// The default method is the agitation sweep with its replay rounds and TSan
// triage, judged by the per-outcome oracle; the TVD method is the opt-in
// alternative and runs no triage.
RunInfo runSingle(const std::string &inputFile, int seed, int runIdx,
                  const PipelineOptions &opts) {
    if (opts.comparison == ComparisonMethod::Agitation)
        return runSingleAgitation(inputFile, seed, runIdx, opts);


    MLIRSetup setup(seed, runIdx, opts.transform, opts.maxApply, opts.model);

    mlir::OwningOpRef<mlir::ModuleOp> originalModule;
    mlir::OwningOpRef<mlir::ModuleOp> moduleToTransform;

    // if unable to apply the requested transforms, return a runInfo with the error and no verdict
    if (!applyTransforms(setup, inputFile, originalModule, moduleToTransform)) {
        setup.runInfo.issues.push_back(failIssue(setup.runInfo.error));
        return setup.runInfo;
    }

    // snapshot both modules before lowering overwrites them
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

    // run the campaign loop: fresh compilations of both variants under the
    // standard configuration, judged by the distributional oracle
    ExecutionResult exec;
    DistributionOracleResult verdict;
    CalibrationResult calibratedVerdict;
    std::string campaignError;
    if (!runCampaignLoop(exec, seed, opts, setup.runInfo.relation, *sourceLLVM,
                         *transformedLLVM, verdict, campaignError,
                         calibratedVerdict)) {
        setup.runInfo.error = campaignError;
        setup.runInfo.issues.push_back(failIssue(campaignError));
        return setup.runInfo;
    }

    populateRunInfoFromExecution(setup.runInfo, exec);
    if (opts.comparison == ComparisonMethod::Calibrated)
        applyCalibrationVerdict(setup.runInfo, calibratedVerdict, opts);
    else
        applyVerdict(setup.runInfo, verdict, opts);
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
