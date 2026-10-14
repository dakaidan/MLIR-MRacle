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
    case DistributionVerdict::Bad:
        return "BAD";
    case DistributionVerdict::Inconclusive:
        break;
    }
    return "INCONCLUSIVE";
}

std::string supportVerdictName(SupportVerdict verdict) {
    switch (verdict) {
    case SupportVerdict::Fail:
        return "FAIL";
    case SupportVerdict::Candidate:
        return "CANDIDATE";
    case SupportVerdict::Ok:
        break;
    }
    return "OK";
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
                     OracleResult &verdict, std::string &error) {
    CampaignOptions campaignOpts;
    campaignOpts.executionsPerCampaign = opts.executionsPerCampaign;
    campaignOpts.teamSize = opts.teamSize;
    campaignOpts.jitOptLevel = kStandardJitOptLevel;

    OracleOptions oracleOpts;
    oracleOpts.relation = relation;
    oracleOpts.maxCampaigns = static_cast<size_t>(std::max(1, opts.campaignsMax));
    oracleOpts.distribution.tvdThreshold = opts.tvdThreshold;
    oracleOpts.distribution.bootstrapReplicates = opts.bootstrapReplicates;
    oracleOpts.distribution.confidenceLevel = opts.confidenceLevel;
    oracleOpts.distribution.thresholdPct = opts.thresholdPct;
    oracleOpts.distribution.bootstrapSeed = static_cast<uint32_t>(seed);
    oracleOpts.distribution.maxRunsPerSide =
        static_cast<int64_t>(std::max(1, opts.campaignsMax)) *
        static_cast<int64_t>(std::max(1, opts.executionsPerCampaign));

    int campaignIndex = 0;
    const int initial = std::max(1, opts.campaignsInitial);
    for (; campaignIndex < initial; ++campaignIndex) {
        if (!runCampaignPair(exec, sourceModule, transformedModule,
                             campaignIndex, campaignSeedFor(seed, campaignIndex),
                             campaignOpts, error))
            return false;
    }

    verdict = oracleCompare(exec, oracleOpts);

    // an inconclusive comparison or an unresolved support candidate buys
    // another batch of independent campaigns, up to the cap. Batching keeps the
    // number of verdicts low while the sample grows towards the rerun budget.
    while (verdict.needsRerun && campaignIndex < opts.campaignsMax) {
        const int batch =
            std::min(initial, opts.campaignsMax - campaignIndex);
        for (int i = 0; i < batch; ++i, ++campaignIndex) {
            if (!runCampaignPair(exec, sourceModule, transformedModule,
                                 campaignIndex,
                                 campaignSeedFor(seed, campaignIndex),
                                 campaignOpts, error))
                return false;
        }
        verdict = oracleCompare(exec, oracleOpts);
    }

    return true;
}

// records the oracle's verdict on the run
void applyVerdict(RunInfo &info, const OracleResult &verdict,
                  const PipelineOptions &opts) {
    info.issues = verdict.compare.issues;

    DistributionSummary summary;
    const std::string status = runStatusString(info);
    summary.verdict = status == "ERROR" ? "BAD" : status == "WARN" ? "WARN" : "OK";
    summary.tvdVerdict = distributionVerdictName(verdict.distribution.verdict);
    summary.tvd = verdict.distribution.tvd;
    summary.ciLow = verdict.distribution.ciLow;
    summary.ciHigh = verdict.distribution.ciHigh;
    summary.tvdThreshold = opts.tvdThreshold;
    summary.confidenceLevel = opts.confidenceLevel;
    summary.bootstrapReplicates = opts.bootstrapReplicates;
    summary.sourceCampaigns =
        static_cast<int64_t>(verdict.distribution.sourceCampaigns);
    summary.transformedCampaigns =
        static_cast<int64_t>(verdict.distribution.transformedCampaigns);
    summary.excludedCampaigns =
        static_cast<int64_t>(verdict.distribution.excludedCampaigns);
    summary.sourceRuns = verdict.distribution.sourceRuns;
    summary.transformedRuns = verdict.distribution.transformedRuns;
    summary.sourceSupport =
        static_cast<int64_t>(verdict.distribution.sourceSupport);
    summary.transformedSupport =
        static_cast<int64_t>(verdict.distribution.transformedSupport);
    summary.sourceEffCategories = verdict.distribution.sourceEffCategories;
    summary.transformedEffCategories =
        verdict.distribution.transformedEffCategories;
    summary.sourceUnseenMass = verdict.distribution.sourceUnseenMass;
    summary.transformedUnseenMass = verdict.distribution.transformedUnseenMass;
    summary.executionsPerCampaign = opts.executionsPerCampaign;
    summary.teamSize = opts.teamSize;
    summary.jitOptLevel = kStandardJitOptLevel;
    info.distribution = std::move(summary);

    for (const auto &violation : verdict.support) {
        SupportViolationSummary record;
        record.outcome = formatOutcome(violation.outcome);
        record.side = violation.transformedOnly ? "transformed" : "source";
        record.count = violation.count;
        record.runs = violation.runs;
        record.rate = violation.rate;
        record.bandRate = violation.bandRate;
        record.significant = violation.significant;
        record.establishable = violation.establishable;
        record.verdict = supportVerdictName(violation.verdict);
        info.supportViolations.push_back(std::move(record));
    }
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
    OracleResult verdict;
    std::string campaignError;
    if (!runCampaignLoop(exec, seed, opts, setup.runInfo.relation, *sourceLLVM,
                         *transformedLLVM, verdict, campaignError)) {
        setup.runInfo.error = campaignError;
        setup.runInfo.issues.push_back(failIssue(campaignError));
        return setup.runInfo;
    }

    populateRunInfoFromExecution(setup.runInfo, exec);
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
