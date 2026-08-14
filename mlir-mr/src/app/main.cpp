#include "mlir-mr/core/core.h"
#include "mlir-mr/io/io.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace {

// writes the artifacts of a single run under <campaignDir>/<status>/run<N>_seed<S>/
void saveArtifacts(const mlir_mr::RunInfo &run, const std::string &status,
                   const std::string &campaignDir) {
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
    mlir_mr::printJson(mlir_mr::runInfoToStatusJson(run), infoOs);
    infoOs << "\n";
    infoOs.flush();
    std::ofstream infoFile((dir / "run_info.json").string());
    infoFile << infoBuf;
}

// writes the .ll artifact of one execution-mode run under
// <campaignDir>/run<N>_seed<S>/
void saveExecutionArtifacts(const mlir_mr::ExecutionRunResult &run,
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

// writes the campaign's result.json
void writeResultJson(const mlir_mr::JsonValue &arr,
                     const std::string &campaignDir) {
    std::filesystem::path logPath =
        std::filesystem::path(campaignDir) / "result.json";
    std::string logBuf;
    llvm::raw_string_ostream logOs(logBuf);
    mlir_mr::printJson(arr, logOs);
    logOs << "\n";
    logOs.flush();
    std::ofstream logFile(logPath.string());
    logFile << logBuf;
}

} // namespace

int main(int argc, char **argv) {
    // Pin the OpenMP runtime to a fixed configuration before any parallel
    // region can initialise it. GOMP_CPU_AFFINITY is read by libgomp,
    // OMP_PROC_BIND by libomp; both are set so either runtime honours the
    // pin. Values are forced so a pre-existing environment cannot change them.
    setenv("OMP_DYNAMIC", "false", 1);
    setenv("OMP_NUM_THREADS", "2", 1);
    setenv("OMP_PROC_BIND", "close", 1);
    setenv("GOMP_CPU_AFFINITY", "0 1", 1);

    mlir_mr::PipelineOptions opts;

    int newArgc = 0;
    for (int i = 0; i < argc; ++i) {
        if (std::strncmp(argv[i], "--seed=", 7) == 0) {
            opts.seed = std::strtol(argv[i] + 7, nullptr, 10);
        } else if (std::strcmp(argv[i], "--run") == 0) {
            opts.straightMode = true;
        } else if (std::strncmp(argv[i], "--reps=", 7) == 0) {
            opts.reps = std::strtol(argv[i] + 7, nullptr, 10);
        } else if (std::strncmp(argv[i], "--iter=", 7) == 0) {
            opts.numRuns = std::strtol(argv[i] + 7, nullptr, 10);
        } else if (std::strncmp(argv[i], "--transform=", 12) == 0) {
            if (!opts.transform.empty())
                opts.transform += ",";
            opts.transform += argv[i] + 12;
        } else if (std::strncmp(argv[i], "--multi=", 8) == 0) {
            opts.multiFolder = argv[i] + 8;
        } else if (std::strncmp(argv[i], "--apply=", 8) == 0) {
            opts.maxApply = std::strtol(argv[i] + 8, nullptr, 10);
        } else if (std::strncmp(argv[i], "--tsan=", 7) == 0) {
            opts.tsanPercent = std::strtol(argv[i] + 7, nullptr, 10);
        } else if (std::strncmp(argv[i], "--campaign-dir=", 15) == 0) {
            opts.campaignDir = argv[i] + 15;
        } else if (std::strncmp(argv[i], "--threshold=", 12) == 0) {
            opts.thresholdPct = std::strtol(argv[i] + 12, nullptr, 10);
        } else if (std::strncmp(argv[i], "--reruns=", 9) == 0) {
            opts.retestReps = std::strtol(argv[i] + 9, nullptr, 10);
        } else if (std::strncmp(argv[i], "--max-runs=", 11) == 0) {
            opts.maxSourceReps = std::strtol(argv[i] + 11, nullptr, 10);
        } else {
            argv[newArgc++] = argv[i];
        }
    }
    argc = newArgc;

    if (opts.multiFolder.empty() && argc < 2) {
        std::cerr << "usage: mlir-mr-opt [--seed=N] "
                     "[--iter=N] [--run] [--reps=N] "
                     "[--transform=NAME[,NAME...]] [--apply=N] "
                     "[--tsan=PERCENT] "
                     "[--multi=FOLDER] [--campaign-dir=PATH] "
                     "[--threshold=PCT] "
                     "[--reruns=N] [--max-runs=N] "
                     "<path-to-mlir-file>\n";
        return 1;
    }

    if (opts.maxApply < 0) {
        std::cerr << "--apply must be >= 0\n";
        return 1;
    }

    if (opts.thresholdPct < 0 || opts.thresholdPct > 100) {
        std::cerr << "--threshold must be between 0 and 100\n";
        return 1;
    }

    if (opts.retestReps <= 0) {
        std::cerr << "--reruns must be > 0\n";
        return 1;
    }

    // Let the retest cap ride along with --reps: a budget above the default
    // cap raises the cap to the requested count instead of aborting, so the
    // source baseline always reaches the requested run count.
    if (opts.maxSourceReps < opts.reps)
        opts.maxSourceReps = opts.reps;

    if (opts.reps <= 0) {
        std::cerr << "--reps must be > 0\n";
        return 1;
    }

    if (!opts.multiFolder.empty())
        opts.inputFile = "";
    else
        opts.inputFile = argv[1];

    if (opts.straightMode) {
        mlir_mr::ExecutionPipelineResult result =
            mlir_mr::runExecutionPipeline(opts);
        for (const auto &run : result.runs)
            saveExecutionArtifacts(run, result.campaignDir);
        mlir_mr::JsonValue arr = mlir_mr::jsonArray();
        for (const auto &run : result.runs)
            mlir_mr::jsonPush(arr, mlir_mr::executionRunToJson(run));
        writeResultJson(arr, result.campaignDir);
        std::cout << result.campaignDir << "\n";
        mlir_mr::shutdownCore();
        return 0;
    }

    mlir_mr::PipelineResult result = mlir_mr::runPipeline(opts);

    std::error_code ec;
    for (const char *sub : {"fail", "warn", "ok"})
        std::filesystem::create_directories(
            std::filesystem::path(result.campaignDir) / sub, ec);

    mlir_mr::JsonValue arr = mlir_mr::jsonArray();
    for (const auto &run : result.runs) {
        mlir_mr::jsonPush(arr, mlir_mr::runInfoToJson(run));
        std::string status = mlir_mr::runStatusString(run);
        if (status == "ERROR")
            saveArtifacts(run, "fail", result.campaignDir);
        else if (status == "WARN")
            saveArtifacts(run, "warn", result.campaignDir);
        else
            saveArtifacts(run, "ok", result.campaignDir);
    }

    writeResultJson(arr, result.campaignDir);
    std::cout << result.campaignDir << "\n";
    mlir_mr::shutdownCore();
    return 0;
}