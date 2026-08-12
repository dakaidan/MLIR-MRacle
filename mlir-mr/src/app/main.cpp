#include "mlir-mr/core/core.h"
#include "mlir-mr/io/io.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <algorithm>

int main(int argc, char **argv) {
    mlir_mr::PipelineOptions opts;

    int newArgc = 0;
    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--print-mlir") == 0) {
            opts.printMLIR = true;
        } else if (std::strncmp(argv[i], "--seed=", 7) == 0) {
            opts.seed = std::strtol(argv[i] + 7, nullptr, 10);
        } else if (std::strncmp(argv[i], "--run=", 6) == 0) {
            opts.runNumber = std::strtol(argv[i] + 6, nullptr, 10);
        } else if (std::strncmp(argv[i], "--runs=", 7) == 0) {
            opts.numRuns = std::strtol(argv[i] + 7, nullptr, 10);
        } else if (std::strncmp(argv[i], "--transform=", 12) == 0) {
            if (!opts.transform.empty())
                opts.transform += ",";
            opts.transform += argv[i] + 12;
        } else if (std::strncmp(argv[i], "--multi=", 8) == 0) {
            opts.multiFolder = argv[i] + 8;
        } else if (std::strncmp(argv[i], "--apply=", 8) == 0) {
            opts.maxApply = std::strtol(argv[i] + 8, nullptr, 10);
        } else if (std::strcmp(argv[i], "--log") == 0) {
            opts.log = true;
        } else if (std::strcmp(argv[i], "--verbose") == 0) {
            opts.verbose = true;
        } else if (std::strncmp(argv[i], "--tsan=", 7) == 0) {
            opts.tsanPercent = std::strtol(argv[i] + 7, nullptr, 10);
        } else if (std::strncmp(argv[i], "--campaign-dir=", 15) == 0) {
            opts.campaignDir = argv[i] + 15;
        } else {
            argv[newArgc++] = argv[i];
        }
    }
    argc = newArgc;

    if (opts.multiFolder.empty() && argc < 2) {
        std::cerr << "usage: mlir-mr-opt [--print-mlir] [--log] [--seed=N] "
                     "[--run=N] [--runs=N] "
                     "[--transform=NAME[,NAME...]] "
                     "[--multi=FOLDER] [--campaign-dir=PATH] "
                     "<path-to-mlir-file>\n";
        return 1;
    }

    if (opts.printMLIR && opts.numRuns > 1) {
        std::cerr << "--print-mlir requires exactly one run (--runs=1 or omit)\n";
        return 1;
    }

    if (opts.maxApply < 0) {
        std::cerr << "--apply must be >= 0\n";
        return 1;
    }

    if (!opts.multiFolder.empty())
        opts.inputFile = "";
    else
        opts.inputFile = argv[1];

    mlir_mr::PipelineResult result = mlir_mr::runPipeline(opts);

    llvm::json::Array arr;
    for (const auto &run : result.runs)
        arr.push_back(mlir_mr::runInfoToJson(run, opts.printMLIR));

    // all outputs are JSON arrays of run info
    llvm::outs() << llvm::json::Value(std::move(arr)) << "\n";
    return 0;
}