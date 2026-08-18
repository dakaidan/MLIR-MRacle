#include "mlir-mr/core/core.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

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
        } else if (std::strcmp(argv[i], "--emit-mlir") == 0) {
            opts.emitMLIR = true;
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
        } else if (std::strncmp(argv[i], "--jit-opt-level=", 16) == 0) {
            opts.jitOptLevel = std::strtol(argv[i] + 16, nullptr, 10);
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
                     "[--iter=N] [--emit-mlir] [--run] [--reps=N] "
                     "[--transform=NAME[,NAME...]] [--apply=N] "
                     "[--tsan=PERCENT] [--jit-opt-level=N] "
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

    if (opts.jitOptLevel < -1 || opts.jitOptLevel > 3) {
        std::cerr << "--jit-opt-level must be between 0 and 3\n";
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

    if (opts.emitMLIR && opts.straightMode) {
        std::cerr << "--emit-mlir and --run cannot be combined\n";
        return 1;
    }
    if (opts.emitMLIR && opts.transform.empty()) {
        std::cerr << "--emit-mlir requires --transform\n";
        return 1;
    }
    if (opts.emitMLIR) {
        // emit mode: --reps is the number of times transformations are
        // picked and applied; --iter remains the repetition count elsewhere
        opts.numRuns = opts.reps;
        mlir_mr::PipelineResult result = mlir_mr::runEmitPipeline(opts);
        std::cout << result.campaignDir << "\n";
        mlir_mr::shutdownCore();
        return 0;
    }

    if (opts.straightMode) {
        mlir_mr::ExecutionPipelineResult result =
            mlir_mr::runExecutionPipeline(opts);
        std::cout << result.campaignDir << "\n";
        mlir_mr::shutdownCore();
        return 0;
    }

    mlir_mr::PipelineResult result = mlir_mr::runPipeline(opts);
    std::cout << result.campaignDir << "\n";
    mlir_mr::shutdownCore();
    return 0;
}