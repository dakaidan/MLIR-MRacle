#include "mlir-mracle/pipeline/execute/execution_pipeline.h"

#include "mlir-mracle/backend/jit/jit.h"
#include "mlir-mracle/context/context.h"
#include "mlir-mracle/execution/execution.h"
#include "mlir-mracle/io/artifacts.h"
#include "mlir-mracle/io/io.h"
#include "mlir-mracle/pipeline/common/pipeline_common.h"

#include <string>
#include <vector>

namespace mlir_mracle {

ExecutionRunResult executeSingle(const std::string &inputFile, int seed,
                                        int runIdx, int reps) {
    ExecutionRunResult result;
    result.runNumber = runIdx;
    result.seed = seed;
    result.file = inputFile;

    mlir::MLIRContext mlirCtx;
    initializeMLIRContext(mlirCtx);

    // parse file
    mlir::OwningOpRef<mlir::ModuleOp> module;
    if (!parseModuleFile(inputFile, mlirCtx, module, result.error))
        return result;

    // lower and translate to LLVM IR, fail if unable
    llvm::LLVMContext llvmCtx;
    std::unique_ptr<llvm::Module> llvmModule;
    if (!lowerAndTranslate(*module, mlirCtx, llvmCtx, "source", nullptr,
                           &result.llvmIR, llvmModule, result.error))
        return result;

    std::string compileError;
    auto fn = compileLLVMModuleToFunction(std::move(llvmModule), &compileError,
                                          false, kLegacyJitOptLevel);
    if (!fn) {
        result.error = "JIT compile error: " + compileError;
        return result;
    }

    // run the compiled function with different thread counts and collect the observed outcomes
    for (int t : kThreadCounts) {
        ObservedOutcomeSet set = collectOutcomeSet(fn, reps, t);
        ExecutionThreadResult tr;
        tr.numThreads = t;
        tr.runs = reps;
        tr.outcomes = std::move(set.outcomes);
        tr.counts = std::move(set.counts);
        result.threadResults.push_back(std::move(tr));
    }
    return result;
}

ExecutionPipelineResult runExecutionPipeline(const PipelineOptions &opts) {
    ExecutionPipelineResult result;
    result.runs.reserve(opts.numRuns);

    createCampaignDir(opts);

    // multi-file handling
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
    }

    // loop over runs, calling executeSingle for each input file and each run, collecting results
    for (int i = 0; i < opts.numRuns; ++i) {
        int runIdx = opts.runNumber + i;
        int runSeed = runSeedFor(opts, runIdx);
        std::string inputFile = pickInputFile(opts, files, runIdx, runSeed);

        ExecutionRunResult run =
            executeSingle(inputFile, runSeed, runIdx, opts.reps);
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

} // namespace mlir_mracle
