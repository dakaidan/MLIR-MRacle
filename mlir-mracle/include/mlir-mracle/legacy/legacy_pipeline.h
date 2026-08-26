#pragma once

#include "mlir-mracle/pipeline/pipeline.h"

namespace mlir_mracle {

// single run mode of the legacy pipeline, returns a RunInfo struct
// with the results of the run.
RunInfo runLegacySingle(const std::string &inputFile, int seed, int runIdx,
                        const std::string &transform, int maxApply,
                        int reps, int retestReps,
                        int maxSourceReps, int thresholdPct);

// core pipeline function for --mode=legacy: runs the metamorphic testing pipeline
// (per-thread-count baseline cache + TSan instrumentation) for the given
// options; returns a PipelineResult struct with the results of all runs
PipelineResult runLegacyPipeline(const PipelineOptions &opts);

} // namespace mlir_mracle
