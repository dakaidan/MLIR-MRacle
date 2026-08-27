#include "cli.h"
#include "mlir-mracle/execution/execution.h"
#include "mlir-mracle/legacy/legacy_pipeline.h"
#include "mlir-mracle/pipeline/emit/emit_pipeline.h"
#include "mlir-mracle/pipeline/execute/execution_pipeline.h"
#include "mlir-mracle/pipeline/pipeline.h"

#include <iostream>

int main(int argc, char **argv) {
    mlir_mracle::applyProcessSettings();

    mlir_mracle::CliOptions cli = mlir_mracle::parsePipelineOptions(argc, argv);

    for (mlir_mracle::PipelineMode mode : cli.modes) {
        mlir_mracle::PipelineOptions opts = cli.pipeline;
        switch (mode) {
        case mlir_mracle::PipelineMode::Emit:
            // emit mode: --reps is the number of times transformations are picked and applied
            // --iter remains the repetition count elsewhere
            opts.numRuns = opts.reps;
            std::cout << mlir_mracle::runEmitPipeline(opts).campaignDir << "\n";
            break;
        case mlir_mracle::PipelineMode::Execute:
            std::cout << mlir_mracle::runExecutionPipeline(opts).campaignDir
                      << "\n";
            break;
        case mlir_mracle::PipelineMode::Legacy:
            std::cout << mlir_mracle::runLegacyPipeline(opts).campaignDir
                      << "\n";
            break;
        case mlir_mracle::PipelineMode::Compare:
            std::cout << mlir_mracle::runPipeline(opts).campaignDir << "\n";
            break;
        }
    }

    mlir_mracle::shutdownPipeline();
    return 0;
}