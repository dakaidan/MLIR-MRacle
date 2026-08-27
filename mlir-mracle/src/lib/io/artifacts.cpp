#include "mlir-mracle/io/artifacts.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <filesystem>
#include <fstream>

namespace mlir_mracle {

void saveRunArtifacts(const RunInfo &run, const std::string &status,
                      const std::string &campaignDir, bool unionFormat) {
    std::filesystem::path dir =
        std::filesystem::path(campaignDir) / status /
        ("run" + std::to_string(run.runNumber) + "_seed" +
         std::to_string(run.seed));
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);

    auto writeIfNonEmpty = [&](const std::string &name,
                               const std::string &content) {
        if (content.empty())
            return;
        std::ofstream os((dir / name).string());
        os << content;
    };

    writeIfNonEmpty("source.mlir", run.sourceMLIR);
    writeIfNonEmpty("transformed.mlir", run.transformedMLIR);
    writeIfNonEmpty("lowered.mlir", run.loweredMLIR);
    writeIfNonEmpty("transformed.ll", run.jitLLVM);
    writeIfNonEmpty("source.ll", run.sourceJitLLVM);
    writeIfNonEmpty("module.bc", run.bitcode);

    std::string infoBuf;
    llvm::raw_string_ostream infoOs(infoBuf);
    printJson(unionFormat ? runInfoToUnionJson(run) : runInfoToStatusJson(run), infoOs);
    infoOs << "\n";
    infoOs.flush();

    std::ofstream infoFile((dir / "run_info.json").string());
    infoFile << infoBuf;
}

void saveExecutionArtifacts(const ExecutionRunResult &run,
                            const std::string &campaignDir) {
    if (run.llvmIR.empty())
        return;

    std::filesystem::path dir =
        std::filesystem::path(campaignDir) /
        ("run" + std::to_string(run.runNumber) + "_seed" +
         std::to_string(run.seed));

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    std::ofstream os((dir / "source.ll").string());
    os << run.llvmIR;
}

void writeResultJson(const JsonValue &arr, const std::string &campaignDir) {
    std::filesystem::path logPath = std::filesystem::path(campaignDir) / "result.json";

    std::string logBuf;
    llvm::raw_string_ostream logOs(logBuf);
    printJson(arr, logOs);
    
    logOs << "\n";
    logOs.flush();

    std::ofstream logFile(logPath.string());
    logFile << logBuf;
}

void createCampaignStatusDirs(const std::string &campaignDir) {
    std::error_code ec;
    for (const char *sub : {"fail", "warn", "ok"})
        std::filesystem::create_directories(
            std::filesystem::path(campaignDir) / sub, ec);
}

} // namespace mlir_mracle
