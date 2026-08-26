#include "mlir-mracle/pipeline/emit/emit_pipeline.h"

#include "mlir-mracle/context/context.h"
#include "mlir-mracle/pipeline/common/pipeline_common.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace mlir_mracle {

// emit mode single run: apply the requested transforms and return the
// resulting MLIR without lowering, JIT, or oracle comparison
RunInfo emitSingle(const std::string &inputFile, int seed,
                          int runIdx, const std::string &transform,
                          int maxApply, const std::string &model) {
    MLIRSetup setup(seed, runIdx, transform, maxApply, model);
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

    for (int i = 0; i < opts.numRuns; ++i) {
        int runIdx = opts.runNumber + i;
        int runSeed = runSeedFor(opts, runIdx);
        std::string inputFile =
            pickInputFile(opts, multiFiles, runIdx, runSeed);
        RunInfo run = emitSingle(inputFile, runSeed, runIdx, opts.transform,
                                 opts.maxApply, opts.model);
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

} // namespace mlir_mracle
