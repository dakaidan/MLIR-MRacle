#include "cli.h"

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

// true when any selected mode uses the distributional protocol
bool usesDistribution(const std::vector<PipelineMode> &modes) {
    for (PipelineMode mode : modes)
        if (mode == PipelineMode::Compare)
            return true;
    return false;
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
            cli.repsExplicit = true;
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
            cli.rerunsExplicit = true;
            cli.pipeline.retestReps = std::strtol(argv[i] + 9, nullptr, 10);
        } else if (std::strncmp(argv[i], "--max-runs=", 11) == 0) {
            cli.maxRunsExplicit = true;
            cli.pipeline.maxSourceReps =
                std::strtol(argv[i] + 11, nullptr, 10);
        } else if (std::strncmp(argv[i], "--campaigns=", 12) == 0) {
            cli.pipeline.campaignsInitial = std::strtol(argv[i] + 12, nullptr, 10);
        } else if (std::strncmp(argv[i], "--max-campaigns=", 16) == 0) {
            cli.pipeline.campaignsMax = std::strtol(argv[i] + 16, nullptr, 10);
        } else if (std::strncmp(argv[i], "--executions-per-campaign=", 26) == 0) {
            cli.pipeline.executionsPerCampaign =
                std::strtol(argv[i] + 26, nullptr, 10);
        } else if (std::strncmp(argv[i], "--team-size=", 12) == 0) {
            cli.pipeline.teamSize = std::strtol(argv[i] + 12, nullptr, 10);
        } else if (std::strncmp(argv[i], "--tvd-threshold=", 16) == 0) {
            cli.pipeline.tvdThreshold = std::strtod(argv[i] + 16, nullptr);
        } else if (std::strncmp(argv[i], "--bootstrap=", 12) == 0) {
            cli.pipeline.bootstrapReplicates =
                std::strtol(argv[i] + 12, nullptr, 10);
        } else if (std::strncmp(argv[i], "--confidence=", 13) == 0) {
            cli.pipeline.confidenceLevel = std::strtod(argv[i] + 13, nullptr);
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
                     "[--campaigns=N] [--max-campaigns=N] "
                     "[--executions-per-campaign=N] [--team-size=N] "
                     "[--tvd-threshold=X] [--bootstrap=N] [--confidence=X] "
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

    // the distributional pipelines measure a campaign budget instead of an
    // execution budget, so their flags are validated on their own terms
    if (usesDistribution(cli.modes)) {
        if (cli.repsExplicit) {
            std::cerr << "compare mode measures campaigns: use "
                         "--executions-per-campaign instead of --reps\n";
            std::exit(1);
        }
        if (cli.rerunsExplicit || cli.maxRunsExplicit) {
            std::cerr << "compare mode samples campaigns: use --campaigns and "
                         "--max-campaigns instead of --reruns and --max-runs\n";
            std::exit(1);
        }
        if (cli.pipeline.campaignsInitial < 2) {
            std::cerr << "--campaigns must be >= 2\n";
            std::exit(1);
        }
        if (cli.pipeline.campaignsMax < cli.pipeline.campaignsInitial) {
            std::cerr << "--max-campaigns must be >= --campaigns\n";
            std::exit(1);
        }
        if (cli.pipeline.executionsPerCampaign <= 0) {
            std::cerr << "--executions-per-campaign must be > 0\n";
            std::exit(1);
        }
        if (cli.pipeline.teamSize <= 0) {
            std::cerr << "--team-size must be > 0\n";
            std::exit(1);
        }
        if (cli.pipeline.tvdThreshold <= 0.0 || cli.pipeline.tvdThreshold > 1.0) {
            std::cerr << "--tvd-threshold must be in (0, 1]\n";
            std::exit(1);
        }
        if (cli.pipeline.bootstrapReplicates <= 0) {
            std::cerr << "--bootstrap must be > 0\n";
            std::exit(1);
        }
        if (cli.pipeline.confidenceLevel <= 0.0 ||
            cli.pipeline.confidenceLevel >= 1.0) {
            std::cerr << "--confidence must be in (0, 1)\n";
            std::exit(1);
        }
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
