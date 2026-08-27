#include "mlir-mracle/app/cli.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <set>

namespace mlir_mracle {
namespace {

bool modeFromName(const std::string &name, PipelineMode &mode) {
    if (name == "emit") {
        mode = PipelineMode::Emit;
    } else if (name == "execution") {
        mode = PipelineMode::Execute;
    } else if (name == "legacy") {
        mode = PipelineMode::Legacy;
    } else if (name == "compare") {
        mode = PipelineMode::Compare;
    } else {
        return false;
    }
    return true;
}

void addMode(std::vector<PipelineMode> &modes, PipelineMode mode) {
    for (PipelineMode m : modes)
        if (m == mode)
            return;
    modes.push_back(mode);
}

void failUnknownMode(const std::string &name) {
    std::cerr << "unknown mode '" << name
              << "' (expected emit, execution, legacy, or compare)\n";
    std::exit(1);
}

} // namespace

CliOptions parsePipelineOptions(int &argc, char **argv) {
    CliOptions cli;

    int newArgc = 0;
    for (int i = 0; i < argc; ++i) {
        if (std::strncmp(argv[i], "--seed=", 7) == 0) {
            cli.pipeline.seed = std::strtol(argv[i] + 7, nullptr, 10);
        } else if (std::strncmp(argv[i], "--mode=", 7) == 0) {
            const char *value = argv[i] + 7;
            if (!*value) {
                failUnknownMode("");
                continue;
            }
            std::string token;
            for (const char *p = value;; ++p) {
                if (*p == ',' || *p == '\0') {
                    PipelineMode mode;
                    if (!modeFromName(token, mode))
                        failUnknownMode(token);
                    addMode(cli.modes, mode);
                    token.clear();
                    if (*p == '\0')
                        break;
                } else {
                    token += *p;
                }
            }
        } else if (std::strncmp(argv[i], "--reps=", 7) == 0) {
            cli.pipeline.reps = std::strtol(argv[i] + 7, nullptr, 10);
        } else if (std::strncmp(argv[i], "--iter=", 7) == 0) {
            cli.pipeline.numRuns = std::strtol(argv[i] + 7, nullptr, 10);
        } else if (std::strncmp(argv[i], "--transform=", 12) == 0) {
            if (!cli.pipeline.transform.empty())
                cli.pipeline.transform += ",";
            cli.pipeline.transform += argv[i] + 12;
        } else if (std::strncmp(argv[i], "--model=", 8) == 0) {
            cli.pipeline.model = argv[i] + 8;
        } else if (std::strncmp(argv[i], "--multi=", 8) == 0) {
            cli.pipeline.multiFolder = argv[i] + 8;
        } else if (std::strncmp(argv[i], "--apply=", 8) == 0) {
            cli.pipeline.maxApply = std::strtol(argv[i] + 8, nullptr, 10);
        } else if (std::strncmp(argv[i], "--campaign-dir=", 15) == 0) {
            cli.pipeline.campaignDir = argv[i] + 15;
        } else if (std::strncmp(argv[i], "--threshold=", 12) == 0) {
            cli.pipeline.thresholdPct = std::strtol(argv[i] + 12, nullptr, 10);
        } else if (std::strncmp(argv[i], "--reruns=", 9) == 0) {
            cli.pipeline.retestReps = std::strtol(argv[i] + 9, nullptr, 10);
        } else if (std::strncmp(argv[i], "--max-runs=", 11) == 0) {
            cli.pipeline.maxSourceReps =
                std::strtol(argv[i] + 11, nullptr, 10);
        } else {
            argv[newArgc++] = argv[i];
        }
    }
    argc = newArgc;

    if (cli.pipeline.multiFolder.empty() && argc < 2) {
        std::cerr << "usage: mlir_mracle-opt [--seed=N] "
                     "[--iter=N] [--reps=N] "
                     "[--transform=NAME[,NAME...]] [--apply=N] "
                     "[--model=NAME] "
                     "[--mode=emit,execution,legacy,compare] "
                     "[--multi=FOLDER] [--campaign-dir=PATH] "
                     "[--threshold=PCT] "
                     "[--reruns=N] [--max-runs=N] "
                     "<path-to-mlir-file>\n";
        std::exit(1);
    }

    if (cli.pipeline.maxApply < 0) {
        std::cerr << "--apply must be >= 0\n";
        std::exit(1);
    }

    if (cli.pipeline.thresholdPct < 0 || cli.pipeline.thresholdPct > 100) {
        std::cerr << "--threshold must be between 0 and 100\n";
        std::exit(1);
    }

    if (cli.pipeline.retestReps <= 0) {
        std::cerr << "--reruns must be > 0\n";
        std::exit(1);
    }

    // Instead of failing, we just set maxSourceReps to reps if it is less than reps
    // Ensures that reps is always the intended number of runs for users
    if (cli.pipeline.maxSourceReps < cli.pipeline.reps)
        cli.pipeline.maxSourceReps = cli.pipeline.reps;

    if (cli.pipeline.reps <= 0) {
        std::cerr << "--reps must be > 0\n";
        std::exit(1);
    }

    if (!cli.pipeline.multiFolder.empty())
        cli.pipeline.inputFile = "";
    else
        cli.pipeline.inputFile = argv[1];

    bool emit = false;
    for (PipelineMode mode : cli.modes)
        if (mode == PipelineMode::Emit)
            emit = true;
    if (emit && cli.pipeline.transform.empty()) {
        std::cerr << "--mode=emit requires --transform\n";
        std::exit(1);
    }

    if (cli.modes.empty())
        cli.modes.push_back(PipelineMode::Compare);

    return cli;
}

} // namespace mlir_mracle
