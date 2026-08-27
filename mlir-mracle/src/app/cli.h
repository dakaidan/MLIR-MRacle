#pragma once

#include "mlir-mracle/pipeline/pipeline.h"

#include <string>
#include <vector>

namespace mlir_mracle {

// pipeline chosen via --mode
enum class PipelineMode {
    Emit,    // --mode=emit: transform and emit MLIR
    Execute, // --mode=execution: execute each file as-is
    Legacy,  // --mode=legacy: legacy thread-group oracle pipeline
    Compare, // --mode=compare: default agitation-sweep oracle pipeline
};

struct CliOptions {
    PipelineOptions pipeline;
    std::vector<PipelineMode> modes;
};

// Parses and validates every binary option flag and returns a CliOptions struct
CliOptions parsePipelineOptions(int &argc, char **argv);

} // namespace mlir_mracle
