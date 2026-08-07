#include "mlir-mr/core/api.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

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
            opts.transform = argv[i] + 12;
        } else if (std::strncmp(argv[i], "--multi=", 8) == 0) {
            opts.multiFolder = argv[i] + 8;
        } else {
            argv[newArgc++] = argv[i];
        }
    }
    argc = newArgc;

    if (opts.multiFolder.empty() && argc < 2) {
        std::cerr << "usage: mlir-mr-opt [--print-mlir] [--seed=N] "
                     "[--run=N] [--runs=N] [--transform=NAME] "
                     "[--multi=FOLDER] <path-to-mlir-file>\n";
        return 1;
    }

    if (opts.printMLIR && opts.numRuns > 1) {
        std::cerr << "--print-mlir requires exactly one run (--runs=1 or omit)\n";
        return 1;
    }

    if (!opts.multiFolder.empty())
        opts.inputFile = "";
    else
        opts.inputFile = argv[1];

    mlir_mr::PipelineResult result = mlir_mr::runPipeline(opts);

    llvm::json::Array arr;
    for (const auto &run : result.runs)
        arr.push_back(run.toJson(opts.printMLIR));

    // all outputs are JSON arrays of run info
    // stdout for outputs, stderr for errors
    llvm::outs() << llvm::json::Value(std::move(arr)) << "\n";
    return 0;
}