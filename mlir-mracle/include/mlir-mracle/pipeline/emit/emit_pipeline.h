#pragma once

#include "mlir-mracle/pipeline/pipeline.h"

namespace mlir_mracle {

// single emit run
// applies the requested transforms to one file and returns the transformed MLIR text
RunInfo emitSingle(const std::string &inputFile, int seed, int runIdx,
                   const std::string &transform, int maxApply,
                   const std::string &model);

// core pipeline function for the default emit oracle
// calls emitSingle for each run and saves the results to the campaign folder
PipelineResult runEmitPipeline(const PipelineOptions &opts);

} // namespace mlir_mracle
