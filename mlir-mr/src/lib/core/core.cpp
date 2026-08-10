#include "mlir-mr/core/api.h"

#include "mlir-mr/backend/jit/jit.h"
#include "mlir-mr/backend/lowering/lowering.h"
#include "mlir-mr/io/io.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <algorithm>

namespace mlir_mr {

// per-campaign log subfolder, created lazily on first logged run
static std::string gCampaignDir;

// number of executions per module per comparison
constexpr int kRuns = 2000;

// a novel outcome (absent from the original's outcome set) that appears in at
// least this fraction of transformed runs is a hard failure; anything rarer is
// a warning. tweak to tighten/loosen the outlier filter.
constexpr double kNovelOutcomeFrequency = 0.05;

// Execution: compiles the module once and records the outcome frequencies of
// every output position over numRuns executions. Returns empty counts and
// sets *error on failure.
static OutcomeCounts ExecuteModule(llvm::Module &module, int numRuns,
                                   std::string *error) {
    OutcomeCounts counts;
    auto fn = compileLLVMModuleToFunction(llvm::CloneModule(module), error);
    if (!fn)
        return counts;
    for (int i = 0; i < numRuns; ++i) {
        auto results = fn();
        if (counts.size() < results.size())
            counts.resize(results.size());
        for (size_t r = 0; r < results.size(); ++r)
            ++counts[r][results[r]];
    }
    return counts;
}

// Comparison: a pure function over two outcome sets sampled in this
// invocation with the same number of runs. Outputs are compared position by
// position: output i of the original is matched against output i of the
// transformed module, never against a different output. A novel outcome in
// the transformed module that appears often enough to rule out sampling noise
// is a hard failure; sparse novel outcomes and vanished outcomes are warnings
// for later triage by the model checker.
static CompareResult CompareOutcomes(const OutcomeCounts &originalCounts,
                                     const OutcomeCounts &transformedCounts,
                                     int numRuns, bool verbose) {
    if (originalCounts.size() != transformedCounts.size())
        return {false, false,
                "result arity mismatch: original produces " +
                    std::to_string(originalCounts.size()) +
                    " outputs, transformed produces " +
                    std::to_string(transformedCounts.size())};

    std::vector<VariableIssue> issues;
    bool identical = true;
    for (size_t i = 0; i < originalCounts.size(); ++i) {
        const auto &orig = originalCounts[i];
        const auto &transf = transformedCounts[i];
        if (orig != transf)
            identical = false;

        std::string label = originalCounts.size() == 1
                                ? "output"
                                : "output " + std::to_string(i);

        bool anyOverlap = false;
        bool anyNovel = false;
        for (auto &[val, _] : transf)
            if (orig.count(val) > 0)
                anyOverlap = true;
            else
                anyNovel = true;

        VariableIssue issue;
        issue.label = label;
        issue.originalSet = formatOutcomeSet(orig);
        issue.transformedSet = formatOutcomeSet(transf);

        // no shared outcome at all: the transformed module is producing
        // fundamentally different results, not explainable by sampling noise
        if (anyNovel && !anyOverlap) {
            issue.disjoint = true;
            issue.hardFail = true;
            issues.push_back(std::move(issue));
            continue;
        }

        // novel outcome frequent enough to be a genuine behavioural change
        for (auto &[val, c] : transf) {
            if (orig.count(val) > 0)
                continue;
            std::string note =
                "novel outcome " + std::to_string(val) + " (" +
                std::to_string(c) + "/" + std::to_string(numRuns) +
                " runs) " +
                (c >= numRuns * kNovelOutcomeFrequency
                     ? "exceeds " +
                           std::to_string(static_cast<int>(
                               kNovelOutcomeFrequency * 100)) +
                           "% of runs"
                     : "below failure threshold");
            if (c >= numRuns * kNovelOutcomeFrequency)
                issue.hardFail = true;
            issue.notes.push_back(note);
        }

        // outcome produced by the original but absent from the transformed
        // module
        for (auto &[val, c] : orig)
            if (transf.count(val) == 0)
                issue.notes.push_back("outcome " + std::to_string(val) +
                                      " disappeared (" + std::to_string(c) +
                                      "/" + std::to_string(numRuns) +
                                      " runs)");

        if (!issue.notes.empty())
            issues.push_back(std::move(issue));
    }

    if (issues.empty()) {
        std::string msg =
            identical
                ? "identical outcome maps over " + std::to_string(numRuns) +
                      " runs"
                : "outcome sets changed over " + std::to_string(numRuns) +
                      " runs";
        return {true, false, msg};
    }

    return renderComparison(numRuns, verbose, issues);
}

// Lazily creates the per-campaign log folder and returns its path.
static const std::string &ensureCampaignDir() {
    if (gCampaignDir.empty()) {
        using namespace std::chrono;
        auto millis = duration_cast<milliseconds>(
            system_clock::now().time_since_epoch()).count();
        gCampaignDir =
            (std::filesystem::path("logs") /
             ("campaign_" + std::to_string(millis))).string();
        std::error_code ec;
        std::filesystem::create_directories(gCampaignDir, ec);
    }
    return gCampaignDir;
}

// Writes the artifacts of a run to logs/<campaign>/run<N>_seed<S>/.
// Only files whose content is available are written; the folder is created
// lazily, so a campaign with nothing logged leaves nothing behind.
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

// single run mode, returns a RunInfo struct with the results of the run
static RunInfo runSingle(const std::string &inputFile, int seed,
                         int runIdx, const std::string &transform,
                         int maxApply, bool printMLIR, bool log,
                         bool verbose) {
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

    // execute both modules the same number of times and compare their outcome
    // sets directly; there is no cached baseline to poison later comparisons
    std::string compileError;

    OutcomeCounts origCounts =
        ExecuteModule(*originalModuleLLVM, kRuns, &compileError);
    if (origCounts.empty()) {
        setup.runInfo.error =
            "JIT compile error (original): " + compileError;
        saveArtifacts();
        return setup.runInfo;
    }

    OutcomeCounts transfCounts =
        ExecuteModule(*moduleToTransformLLVM, kRuns, &compileError);
    if (transfCounts.empty()) {
        setup.runInfo.error =
            "JIT compile error (transformed): " + compileError;
        saveArtifacts();
        return setup.runInfo;
    }

    CompareResult cmp =
        CompareOutcomes(origCounts, transfCounts, kRuns, verbose);

    if (!cmp.ok)
        setup.runInfo.error = cmp.message;
    if (cmp.warn)
        setup.runInfo.warn = cmp.message;

    // failures are always logged; successful runs only with --log
    if (!setup.runInfo.error.empty() || log)
        saveArtifacts();

    return setup.runInfo;
}

// core pipeline function, runs the metamorphic testing pipeline for the given options and returns a PipelineResult struct with the results of all runs
PipelineResult runPipeline(const PipelineOptions &opts) {
    PipelineResult result;
    result.runs.reserve(opts.numRuns);

    // each pipeline invocation gets its own log folder; created lazily
    // on the first logged run
    gCampaignDir.clear();

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
        // here, so stdout stays machine-parseable
        RunInfo info = runSingle(inputFile, runSeed, runIdx,
                                 opts.transform, opts.maxApply,
                                 opts.printMLIR, opts.log, opts.verbose);

        result.runs.push_back(std::move(info));
    }

    return result;
}

} // namespace mlir_mr
