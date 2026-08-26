#pragma once

#include "mlir-mracle/pipeline/pipeline.h"

namespace mlir_mracle {

// emit mode single run: apply the requested transforms and return the
// resulting MLIR without lowering, JIT, or oracle comparison
RunInfo emitSingle(const std::string &inputFile, int seed, int runIdx,
                   const std::string &transform, int maxApply,
                   const std::string &model);

// generator mode for --mode=emit: applies the requested transforms to one
// file or random files from a folder and returns the transformed MLIR text.
// Each run is written to the output folder as it completes; there is no
// execution state.
PipelineResult runEmitPipeline(const PipelineOptions &opts);

} // namespace mlir_mracle
