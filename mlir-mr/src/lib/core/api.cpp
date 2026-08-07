#include "mlir-mr/core/api.h"

#include "mlir-mr/backend/lowering/lowering.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/JSON.h"

#include <filesystem>
#include <random>
#include <set>
#include <string>

namespace mlir_mr {

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

// single run mode, returns a RunInfo struct with the results of the run
static RunInfo runSingle(const std::string &inputFile, int seed,
                         int runIdx, const std::string &transform,
                         bool printMLIR) {
    MLIRSetup setup(seed, runIdx, transform);
    setup.runInfo.file = inputFile;

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
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceFile<mlir::ModuleOp>(inputFile, &setup.mlirContext);
    if (!module)
        return setup.runInfo;
    setup.runInfo.error.clear();

    // similar to above but with the pass pipeline, if it fails return the RunInfo with the error, otherwise clear the error
    setup.runInfo.error = "pass pipeline failed";
    if (mlir::failed(setup.pm.run(*module)))
        return setup.runInfo;
    setup.runInfo.error.clear();

    // if requested, print the MLIR module to a string and store it in the RunInfo struct
    // this is not found on the multiple run mode, as it is only useful for debugging a single run
    if (printMLIR) {
        std::string buf;
        llvm::raw_string_ostream os(buf);
        module->print(os);
        os.flush();
        setup.runInfo.mlirOutput = buf;
    }

    // if lowering to LLVM fails, return the RunInfo with the error, otherwise clear the error
    setup.runInfo.error = "lowering to LLVM failed";
    if (mlir::failed(mlir_mr::lowerToLLVM(*module, &setup.mlirContext)))
        return setup.runInfo;
    setup.runInfo.error.clear();

    return setup.runInfo;
}

// core pipeline function, runs the metamorphic testing pipeline for the given options and returns a PipelineResult struct with the results of all runs
PipelineResult runPipeline(const PipelineOptions &opts) {
    PipelineResult result;
    result.runs.reserve(opts.numRuns);

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

        // run single run and collect the RunInfo, if there is an error print it to stderr, otherwise add it to the result
        RunInfo info = runSingle(inputFile, runSeed, runIdx,
                                 opts.transform, opts.printMLIR);

        // if there is an error, print it to stderr with the run number, seed, and file name
        if (!info.error.empty()) {
            llvm::errs() << "[ERROR] run=" << info.runNumber
                         << " seed=" << info.seed
                         << " file=" << info.file
                         << ": " << info.error << "\n";
            llvm::errs().flush();
        }

        // otherwise add the RunInfo to the result
        result.runs.push_back(std::move(info));
    }

    return result;
}

} // namespace mlir_mr
