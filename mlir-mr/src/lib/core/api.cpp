#include "mlir-mr/core/api.h"

#include "mlir-mr/backend/jit/jit.h"
#include "mlir-mr/backend/lowering/lowering.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/JSON.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>
#include <unordered_map>

namespace mlir_mr {

// Result of comparing the two compiled modules; ok=false signals an error or a
// behavioural mismatch, message carries the human-readable detail.
struct CompareResult {
    bool ok = true;
    std::string message;
};

// Chi-squared statistic for two equal-sized groups over a shared,
// compacted category index [0, numCategories).
static double chiSquaredFromCounts(const std::vector<int> &countA,
                                    const std::vector<int> &countB) {
    double chi2 = 0.0;
    for (size_t c = 0; c < countA.size(); ++c) {
        double total = countA[c] + countB[c];
        if (total == 0.0)
            continue;
        double expected = total / 2.0; // equal group sizes
        double dA = countA[c] - expected;
        double dB = countB[c] - expected;
        chi2 += (dA * dA) / expected + (dB * dB) / expected;
    }
    return chi2;
}

// Monte-Carlo permutation test
static double permutationPValue(std::vector<int> combinedIdx, int numCategories,
                                 size_t n, double observedChi2,
                                 int numPermutations, std::mt19937 &rng) {
    std::vector<int> countA(numCategories), countB(numCategories);
    int atLeastAsExtreme = 0;
    for (int p = 0; p < numPermutations; ++p) {
        std::shuffle(combinedIdx.begin(), combinedIdx.end(), rng);
        std::fill(countA.begin(), countA.end(), 0);
        std::fill(countB.begin(), countB.end(), 0);
        for (size_t i = 0; i < n; ++i)
            ++countA[combinedIdx[i]];
        for (size_t i = n; i < combinedIdx.size(); ++i)
            ++countB[combinedIdx[i]];
        double chi2 = chiSquaredFromCounts(countA, countB);
        if (chi2 >= observedChi2 - 1e-9)
            ++atLeastAsExtreme;
    }
    // +1 smoothing: standard for MC permutation p-values, avoids a hard 0.
    return static_cast<double>(atLeastAsExtreme + 1) / (numPermutations + 1);
}

static CompareResult ExecuteAndCompareModules(llvm::Module *module,
                                              llvm::Module *transformedModule) {
    constexpr int kRuns = 300;
    constexpr int kNumPermutations = 100000;
    constexpr double kAlpha = 0.0001;

    std::string compileError;

    auto origFn = compileLLVMModuleToFunction(
        llvm::CloneModule(*module), &compileError);
    if (!origFn)
        return {false, "JIT compile error (original): " + compileError};

    auto transfFn = compileLLVMModuleToFunction(
        llvm::CloneModule(*transformedModule), &compileError);
    if (!transfFn)
        return {false, "JIT compile error (transformed): " + compileError};

    std::map<int32_t, int> originalCounts, transformedCounts;
    std::vector<int32_t> combined;
    combined.reserve(kRuns * 2);
    for (int i = 0; i < kRuns; ++i) {
        int32_t o = origFn();
        int32_t t = transfFn();
        ++originalCounts[o];
        ++transformedCounts[t];
        combined.push_back(o);
        combined.push_back(t);
    }

    if (originalCounts == transformedCounts)
        return {true, "match: identical outcome maps over " +
                          std::to_string(kRuns) + " runs"};

    // Permutation test for distribution drift (maps differ but may still
    // be statistically equivalent)
    std::unordered_map<int32_t, int> categoryIndex;
    auto index = [&](int32_t v) -> int {
        auto it = categoryIndex.find(v);
        if (it != categoryIndex.end())
            return it->second;
        int id = static_cast<int>(categoryIndex.size());
        categoryIndex.emplace(v, id);
        return id;
    };

    for (auto &[val, _] : originalCounts) index(val);
    for (auto &[val, _] : transformedCounts) index(val);
    int numCategories = static_cast<int>(categoryIndex.size());

    std::vector<int> countA(numCategories), countB(numCategories);
    for (auto &[val, c] : originalCounts) countA[index(val)] = c;
    for (auto &[val, c] : transformedCounts) countB[index(val)] = c;

    double observedChi2 = chiSquaredFromCounts(countA, countB);

    std::vector<int> combinedIdx;
    combinedIdx.reserve(combined.size());
    for (int32_t v : combined)
        combinedIdx.push_back(index(v));

    std::mt19937 rng(std::random_device{}());
    double pValue = permutationPValue(std::move(combinedIdx), numCategories,
                                       kRuns, observedChi2, kNumPermutations, rng);

    if (pValue < kAlpha)
        return {false, "WARN: distribution differs (chi2=" +
                           std::to_string(observedChi2) + ", p=" +
                           std::to_string(pValue) + " < " +
                           std::to_string(kAlpha) + ")"};

    return {true, "match: distributions equivalent over " +
                      std::to_string(kRuns) + " runs (permutation p=" +
                      std::to_string(pValue) + ")"};
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
                         int maxApply, bool printMLIR) {
    MLIRSetup setup(seed, runIdx, transform, maxApply);
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

    // if requested, print the MLIR module to a string and store it in the RunInfo struct
    // this is not found on the multiple run mode, as it is only useful for debugging a single run
    if (printMLIR) {
        std::string buf;
        llvm::raw_string_ostream os(buf);
        moduleToTransform->print(os);
        os.flush();
        setup.runInfo.mlirOutput = buf;
    }

    // lower the source module to LLVM IR
    setup.runInfo.error = "lowering of source module to LLVM failed";
    if (mlir::failed(mlir_mr::lowerToLLVM(*originalModule, &setup.mlirContext)))
        return setup.runInfo;
    setup.runInfo.error.clear();

    setup.runInfo.error = "translation of source module to LLVM IR failed";
    std::unique_ptr<llvm::Module> originalModuleLLVM =
        mlir::translateModuleToLLVMIR(*originalModule, setup.llvmContext);
    if (!originalModuleLLVM)
        return setup.runInfo;
    setup.runInfo.error.clear();

    // lower the transformed module to LLVM IR
    setup.runInfo.error = "lowering of transformed module to LLVM failed";
    if (mlir::failed(mlir_mr::lowerToLLVM(*moduleToTransform, &setup.mlirContext)))
        return setup.runInfo;
    setup.runInfo.error.clear();

    setup.runInfo.error = "translation of transformed module to LLVM IR failed";
    std::unique_ptr<llvm::Module> moduleToTransformLLVM =
        mlir::translateModuleToLLVMIR(*moduleToTransform, setup.llvmContext);
    if (!moduleToTransformLLVM)
        return setup.runInfo;
    setup.runInfo.error.clear();

    // execute the two LLVM IR modules and compare the results, storing an error only if the comparison fails
    CompareResult cmp = ExecuteAndCompareModules(originalModuleLLVM.get(),
                                                 moduleToTransformLLVM.get());
    if (!cmp.ok)
        setup.runInfo.error = cmp.message;

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
                                 opts.transform, opts.maxApply,
                                 opts.printMLIR);

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
