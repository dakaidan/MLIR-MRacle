#include "mlir-mracle/pipeline/common/pipeline_common.h"

#include "mlir-mracle/backend/lowering/lowering.h"
#include "mlir-mracle/io/io.h"
#include "mlir-mracle/passes/MetamorphicPass.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace mlir_mracle {

// per-campaign log folder; runs are written into it as they complete
std::string gCampaignDir;

// maps a run's status ("OK"/"WARN"/"ERROR") onto its campaign subfolder
// ("ok"/"warn"/"fail")
const char *statusDirFor(const RunInfo &info) {
    std::string status = runStatusString(info);
    if (status == "ERROR")
        return "fail";
    if (status == "WARN")
        return "warn";
    return "ok";
}

// creates (or reuses) the campaign folder used by both pipelines
void createCampaignDir(const PipelineOptions &opts) {
    if (!opts.campaignDir.empty()) {
        gCampaignDir = opts.campaignDir;
    } else {
        using namespace std::chrono;
        auto millis = duration_cast<milliseconds>(
            system_clock::now().time_since_epoch()).count();
        gCampaignDir =
            (std::filesystem::path("results") /
             ("campaign_" + std::to_string(millis))).string();
    }
    std::error_code ec;
    std::filesystem::create_directories(gCampaignDir, ec);
}

// helper function for multi mode, collects all .mlir files in the given folder
std::vector<std::string> collectMLIRFiles(const std::string &folder) {
    std::vector<std::string> files;
    std::error_code ec;
    for (const auto &entry :
         std::filesystem::directory_iterator(folder, ec)) {
        if (entry.is_regular_file() && entry.path().extension() == ".mlir")
            files.push_back(entry.path().string());
    }
    std::sort(files.begin(), files.end());
    return files;
}

// picks a random .mlir file for a run
std::string pickInputFile(const PipelineOptions &opts,
                          const std::vector<std::string> &multiFiles,
                          int runIdx, int runSeed) {
    if (multiFiles.empty())
        return opts.inputFile;
    uint32_t fileSeed = static_cast<uint32_t>(runSeed) +
                        static_cast<uint32_t>(runIdx) * 0x9e3779b9u;
    std::mt19937 fileRng(fileSeed);
    std::uniform_int_distribution<size_t> dist(0, multiFiles.size() - 1);
    return multiFiles[dist(fileRng)];
}

// if the user provided a seed, use it, otherwise random seed
int runSeedFor(const PipelineOptions &opts, int runIdx) {
    if (opts.seed >= 0)
        return opts.seed;
    static const uint32_t base = [] {
        std::random_device rd;
        return rd();
    }();
    uint32_t h = base + static_cast<uint32_t>(runIdx) * 2654435761u;
    return static_cast<int>(h & 0x7FFFFFFFu);
}

// parses MLIR source file
bool parseModuleFile(const std::string &file, mlir::MLIRContext &ctx,
                     mlir::OwningOpRef<mlir::ModuleOp> &module,
                     std::string &error) {
    mlir::ScopedDiagnosticHandler diagHandler(
        &ctx, [&](mlir::Diagnostic &diag) {
            if (!error.empty())
                error += "; ";
            error += diag.str();
            return mlir::success();
        });
    error = "parse error";
    module = mlir::parseSourceFile<mlir::ModuleOp>(file, &ctx);
    if (!module)
        return false;
    error.clear();
    return true;
}

// applies the requested transforms to MLIR module
bool applyTransforms(MLIRSetup &setup, const std::string &inputFile,
                     mlir::OwningOpRef<mlir::ModuleOp> &originalModule,
                     mlir::OwningOpRef<mlir::ModuleOp> &transformedModule) {
    setup.runInfo.file = inputFile;
    if (!parseModuleFile(inputFile, setup.mlirContext, originalModule,
                         setup.runInfo.error))
        return false;
    setup.runInfo.sourceMLIR = dumpMLIR(*originalModule);
    transformedModule = mlir::OwningOpRef<mlir::ModuleOp>(
        mlir::ModuleOp(originalModule->clone()));

    mlir::ScopedDiagnosticHandler diagHandler(
        &setup.mlirContext, [&](mlir::Diagnostic &diag) {
            if (!setup.runInfo.error.empty())
                setup.runInfo.error += "; ";
            setup.runInfo.error += diag.str();
            return mlir::success();
        });
    setup.runInfo.error = "pass pipeline failed";
    if (mlir::failed(setup.pm.run(*transformedModule)))
        return false;
    setup.runInfo.error.clear();
    setup.runInfo.transformedMLIR = dumpMLIR(*transformedModule);
    return true;
}

// Applies random jitter to MLIR module for agitation
bool applyJitter(mlir::MLIRContext &ctx, mlir::ModuleOp module, int seed,
                 std::string &error) {
    mlir::ScopedDiagnosticHandler diagHandler(
        &ctx, [&](mlir::Diagnostic &diag) {
            if (!error.empty())
                error += "; ";
            error += diag.str();
            return mlir::success();
        });
    error = "jitter pass failed";
    mlir::PassManager pm(&ctx, mlir::ModuleOp::getOperationName());
    pm.addPass(mlir::createMetamorphicPass(seed, nullptr, "insert-jitter", 1));
    if (mlir::failed(pm.run(module)))
        return false;
    error.clear();
    return true;
}

// lowers a module to the LLVM dialect and translates it to LLVM IR
bool lowerAndTranslate(mlir::ModuleOp module, mlir::MLIRContext &mlirCtx,
                       llvm::LLVMContext &llvmCtx, const std::string &label,
                       std::string *loweredMLIR, std::string *llvmIR,
                       std::unique_ptr<llvm::Module> &llvmModule,
                       std::string &error) {
    mlir::ScopedDiagnosticHandler diagHandler(
        &mlirCtx, [&](mlir::Diagnostic &diag) {
            if (!error.empty())
                error += "; ";
            error += diag.str();
            return mlir::success();
        });
    error = "lowering of " + label + " module to LLVM failed";
    if (mlir::failed(mlir_mracle::lowerToLLVM(module, &mlirCtx)))
        return false;
    error.clear();
    if (loweredMLIR)
        *loweredMLIR = dumpMLIR(module);
    error = "translation of " + label + " module to LLVM IR failed";
    llvmModule = mlir::translateModuleToLLVMIR(module, llvmCtx);
    if (!llvmModule)
        return false;
    error.clear();
    if (llvmIR)
        *llvmIR = dumpLLVM(*llvmModule);
    return true;
}

} // namespace mlir_mracle
